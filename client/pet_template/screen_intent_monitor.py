import argparse
import base64
import ctypes
import io
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mss
import requests
from PIL import Image, ImageChops


DEFAULT_OPENAI_MODEL = "gpt-5.6"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MAX_PATH_CHARS = 32768


@dataclass
class WindowInfo:
    title: str
    process_name: str
    process_path: str


@dataclass
class IntentResult:
    timestamp: str
    provider: str
    model: str
    idle_seconds: float
    window: WindowInfo
    activity: str
    intent: str
    confidence: float
    should_interrupt: bool
    sensitive_visible: bool
    summary: str
    raw_text: str = ""


def idle_seconds() -> float:
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
    return max(0.0, millis / 1000.0)


def foreground_window_info() -> WindowInfo:
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return WindowInfo(title="", process_name="", process_path="")

    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    title = buffer.value

    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    process_path = query_process_path(pid.value)
    process_name = Path(process_path).name if process_path else ""

    return WindowInfo(title=title, process_name=process_name, process_path=process_path)


def query_process_path(pid: int) -> str:
    if not pid:
        return ""

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""

    try:
        size = ctypes.c_ulong(MAX_PATH_CHARS)
        buffer = ctypes.create_unicode_buffer(MAX_PATH_CHARS)
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
        return buffer.value if ok else ""
    finally:
        kernel32.CloseHandle(handle)


def capture_screen(max_width: int) -> Image.Image:
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.rgb)

    if max_width > 0 and image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)

    return image


def jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def changed_enough(previous: Image.Image | None, current: Image.Image, threshold: int) -> bool:
    if previous is None:
        return True

    prev = previous.resize((320, 180)).convert("L")
    curr = current.resize((320, 180)).convert("L")
    diff = ImageChops.difference(prev, curr)
    score = sum(diff.getdata())
    return score >= threshold


def choose_provider(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "heuristic"


def build_prompt(window: WindowInfo, previous_summary: str) -> str:
    return (
        "你是一个只用于用户本人电脑的屏幕活动理解模块。"
        "根据截图、前台窗口标题和程序名，判断用户现在最可能在做什么。\n"
        "要求：\n"
        "1. 只输出 JSON，不要输出 Markdown。\n"
        "2. 不要逐字抄写私人聊天、密钥、密码、账号、地址等敏感内容。\n"
        "3. 如果屏幕含有敏感内容，把 sensitive_visible 设为 true，但 summary 只做概括。\n"
        "4. should_interrupt 表示桌面助手现在是否适合主动打扰用户。\n"
        "5. confidence 是 0 到 1 的数字。\n\n"
        "JSON schema:\n"
        "{"
        "\"activity\":\"coding|reading|chatting|watching_video|gaming|writing|browsing|meeting|idle|unknown\","
        "\"intent\":\"一句话说明用户当前意图\","
        "\"confidence\":0.0,"
        "\"should_interrupt\":false,"
        "\"sensitive_visible\":false,"
        "\"summary\":\"一句话中文概括\""
        "}\n\n"
        f"前台程序: {window.process_name or 'unknown'}\n"
        f"前台标题: {window.title or 'unknown'}\n"
        f"上一次摘要: {previous_summary or '无'}"
    )


def analyze_with_openai(
    image_data: bytes,
    prompt: str,
    model: str,
    timeout: int,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode("ascii")
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "max_output_tokens": 400,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return extract_openai_text(body)


def extract_openai_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    chunks: list[str] = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def analyze_with_gemini(
    image_data: bytes,
    prompt: str,
    model: str,
    timeout: int,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

    payload = {
        "model": model,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": base64.b64encode(image_data).decode("ascii"),
                "mime_type": "image/jpeg",
            },
        ],
    }
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    response = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return extract_gemini_text(response.json())


def extract_gemini_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    chunks: list[str] = []
    for candidate in body.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def analyze_heuristically(window: WindowInfo) -> str:
    text = f"{window.process_name} {window.title}".lower()
    if any(name in text for name in ["code.exe", "pycharm", "visual studio", ".py", ".js", ".ts"]):
        activity = "coding"
        intent = "用户可能正在编写或调试代码。"
    elif any(name in text for name in ["chrome", "edge", "firefox", "browser"]):
        activity = "browsing"
        intent = "用户可能正在浏览网页。"
    elif any(name in text for name in ["wechat", "qq", "telegram", "discord"]):
        activity = "chatting"
        intent = "用户可能正在聊天。"
    elif any(name in text for name in ["steam", "game", "unity", "unreal"]):
        activity = "gaming"
        intent = "用户可能正在玩游戏。"
    else:
        activity = "unknown"
        intent = "仅凭窗口信息无法可靠判断用户意图。"

    return json.dumps(
        {
            "activity": activity,
            "intent": intent,
            "confidence": 0.35,
            "should_interrupt": False,
            "sensitive_visible": False,
            "summary": intent,
        },
        ensure_ascii=False,
    )


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    return {
        "activity": str(data.get("activity", "unknown")),
        "intent": str(data.get("intent", "")),
        "confidence": clamp_float(data.get("confidence", 0.0)),
        "should_interrupt": parse_bool(data.get("should_interrupt", False)),
        "sensitive_visible": parse_bool(data.get("sensitive_visible", False)),
        "summary": str(data.get("summary", "")),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


def analyze_intent(
    provider: str,
    model: str,
    image_data: bytes,
    window: WindowInfo,
    previous_summary: str,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    prompt = build_prompt(window, previous_summary)

    if provider == "openai":
        raw_text = analyze_with_openai(image_data, prompt, model, timeout)
    elif provider == "gemini":
        raw_text = analyze_with_gemini(image_data, prompt, model, timeout)
    elif provider == "heuristic":
        raw_text = analyze_heuristically(window)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    return parse_model_json(raw_text), raw_text


def print_result(result: IntentResult) -> None:
    marker = "interrupt" if result.should_interrupt else "quiet"
    sensitive = " sensitive" if result.sensitive_visible else ""
    print(
        f"[{result.timestamp}] {result.activity} {result.confidence:.2f} "
        f"{marker}{sensitive} | {result.intent}"
    )
    if result.summary and result.summary != result.intent:
        print(f"  summary: {result.summary}")
    if result.window.process_name or result.window.title:
        print(f"  window: {result.window.process_name} | {result.window.title}")


def append_jsonl(path: Path, result: IntentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def save_debug_screenshot(path: Path, image_data: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    (path / name).write_bytes(image_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the local screen and print a low-frequency intent summary."
    )
    parser.add_argument("--provider", choices=["auto", "openai", "gemini", "heuristic"], default="auto")
    parser.add_argument("--model", default="")
    parser.add_argument("--active-check-interval", type=float, default=3.0)
    parser.add_argument("--analyze-interval", type=float, default=30.0)
    parser.add_argument("--idle-threshold", type=float, default=60.0)
    parser.add_argument("--idle-sleep", type=float, default=15.0)
    parser.add_argument("--change-threshold", type=int, default=5000)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--save-screenshots", default="")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


args = parse_args()
provider = choose_provider(args.provider)
model = args.model or (
    DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_GEMINI_MODEL if provider == "gemini" else "heuristic"
)
log_file = Path(args.log_file) if args.log_file else None
screenshot_dir = Path(args.save_screenshots) if args.save_screenshots else None

print(f"screen intent monitor started: provider={provider}, model={model}")
print("Press Ctrl+C to stop. Screenshots are not saved unless --save-screenshots is set.")

previous_image: Image.Image | None = None
previous_summary = ""
next_analysis_at = 0.0
idle_reported = False

def check() -> int:
    global idle_reported,previous_image,next_analysis_at,previous_summary
    idle = idle_seconds()
    if idle >= args.idle_threshold:
        if not idle_reported:
            print(f"idle for {idle:.0f}s; sleeping")
            idle_reported = True
        if args.once:
            return 0
        time.sleep(args.idle_sleep)
        return -1

    if idle_reported:
        print("activity resumed")
        idle_reported = False
        previous_image = None
        next_analysis_at = 0.0

    now = time.monotonic()
    image = capture_screen(args.max_width)
    has_changed = changed_enough(previous_image, image, args.change_threshold)

    if has_changed and now >= next_analysis_at:
        window = foreground_window_info()
        image_data = jpeg_bytes(image, args.jpeg_quality)
        try:
            parsed, raw_text = analyze_intent(
                provider=provider,
                model=model,
                image_data=image_data,
                window=window,
                previous_summary=previous_summary,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"analysis failed: {exc}", file=sys.stderr)
            if args.once:
                return 1
            next_analysis_at = now + args.analyze_interval
            time.sleep(args.active_check_interval)
            return -1

        result = IntentResult(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            provider=provider,
            model=model,
            idle_seconds=round(idle, 2),
            window=window,
            raw_text=raw_text,
            **parsed,
        )
        previous_summary = result.summary or result.intent
        previous_image = image
        next_analysis_at = now + args.analyze_interval

        print_result(result)
        if log_file:
            append_jsonl(log_file, result)
        if screenshot_dir:
            save_debug_screenshot(screenshot_dir, image_data)
    elif has_changed:
        previous_image = image

    if args.once:
        return 0
    time.sleep(args.active_check_interval)

# if __name__ == "__main__":
#     raise SystemExit(main())
