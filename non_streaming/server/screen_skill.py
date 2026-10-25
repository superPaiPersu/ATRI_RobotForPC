import base64
import ctypes
from contextvars import ContextVar
import io
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image

try:
    import mss
except ImportError:
    mss = None


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


DEFAULT_OPENAI_MODEL = "gpt-5.6"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_WIDTH = max(320, env_int("SCREEN_SKILL_MAX_WIDTH", 720))
DEFAULT_JPEG_QUALITY = min(95, max(30, env_int("SCREEN_SKILL_JPEG_QUALITY", 58)))
DEFAULT_TIMEOUT_SECONDS = env_float("SCREEN_SKILL_TIMEOUT_SECONDS", 12)
DEFAULT_IDLE_THRESHOLD_SECONDS = 60
DEFAULT_VISION_MAX_REQUESTS_PER_MINUTE = 6
DEFAULT_ALLOW_SERVER_CAPTURE = os.getenv("SCREEN_SKILL_ALLOW_SERVER_CAPTURE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = env_int("SCREEN_SKILL_GEMINI_MAX_OUTPUT_TOKENS", 1536)
DEFAULT_GEMINI_THINKING_LEVEL = os.getenv("SCREEN_SKILL_GEMINI_THINKING_LEVEL", "minimal").strip().lower()
DEFAULT_GEMINI_MEDIA_RESOLUTION = os.getenv(
    "SCREEN_SKILL_GEMINI_MEDIA_RESOLUTION",
    "MEDIA_RESOLUTION_MEDIUM",
).strip().upper()
SCREEN_ANALYSIS_ACTIVITIES = {
    "coding",
    "reading",
    "chatting",
    "watching_video",
    "gaming",
    "writing",
    "browsing",
    "meeting",
    "idle",
    "unknown",
}
GEMINI_SCREEN_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "activity": {
            "type": "STRING",
            "enum": sorted(SCREEN_ANALYSIS_ACTIVITIES),
        },
        "intent": {"type": "STRING"},
        "confidence": {"type": "NUMBER", "minimum": 0, "maximum": 1},
        "should_interrupt": {"type": "BOOLEAN"},
        "sensitive_visible": {"type": "BOOLEAN"},
        "summary": {"type": "STRING"},
    },
    "required": [
        "activity",
        "intent",
        "confidence",
        "should_interrupt",
        "sensitive_visible",
        "summary",
    ],
}


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MAX_PATH_CHARS = 32768
_IS_WINDOWS = os.name == "nt" and hasattr(ctypes, "windll")

_vision_rate_lock = threading.Lock()
_vision_request_times: list[float] = []
_screen_capture_request_var: ContextVar[
    Callable[[dict[str, Any]], dict[str, Any]] | None
] = ContextVar("screen_capture_request", default=None)


RETRYABLE_GEMINI_KEY_STATUSES = {401, 403, 429, 503}


@dataclass
class VisionModelResult:
    text: str
    key_index: int | None = None
    attempts: list[dict[str, Any]] | None = None


class GeminiKeyPoolExhaustedError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]], status: int | None):
        super().__init__(message)
        self.attempts = attempts
        self.status = status


class VisionRateLimitError(RuntimeError):
    def __init__(
        self,
        retry_after_seconds: float,
        max_requests_per_minute: int,
        attempts: list[dict[str, Any]] | None = None,
    ):
        super().__init__("Screen vision request rate limit reached.")
        self.retry_after_seconds = retry_after_seconds
        self.max_requests_per_minute = max_requests_per_minute
        self.attempts = attempts or []
        self.status = 429


@dataclass
class WindowInfo:
    title: str
    process_name: str
    process_path: str


def set_screen_capture_request(
    requester: Callable[[dict[str, Any]], dict[str, Any]] | None,
):
    return _screen_capture_request_var.set(requester)


def reset_screen_capture_request(token) -> None:
    _screen_capture_request_var.reset(token)


def _request_client_screen_capture(payload: dict[str, Any]) -> dict[str, Any] | None:
    requester = _screen_capture_request_var.get()
    if requester is None:
        return None
    response = requester(payload)
    if not isinstance(response, dict):
        raise RuntimeError("Client screen capture returned an invalid response.")
    if response.get("error"):
        raise RuntimeError(str(response.get("error")))
    return response


def _decode_client_capture_response(response: dict[str, Any]) -> tuple[bytes, dict[str, Any], float]:
    image_b64 = (
        response.get("image_base64")
        or response.get("image_b64")
        or response.get("jpeg_base64")
    )
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise RuntimeError("Client screen capture response did not include image data.")

    image_data = base64.b64decode(image_b64)
    image_width = response.get("image_width")
    image_height = response.get("image_height")
    try:
        image_width = int(image_width)
        image_height = int(image_height)
    except (TypeError, ValueError):
        image_width = image_height = 0

    if image_width <= 0 or image_height <= 0:
        with Image.open(io.BytesIO(image_data)) as image:
            image_width, image_height = image.size

    freshness = {
        "capture_started_at": response.get("capture_started_at") or datetime.now().isoformat(timespec="milliseconds"),
        "capture_finished_at": response.get("capture_finished_at") or datetime.now().isoformat(timespec="milliseconds"),
        "capture_latency_seconds": response.get("capture_latency_seconds"),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "jpeg_bytes": len(image_data),
        "is_live_stream": False,
        "capture_source": response.get("capture_source") or "client",
    }
    idle_seconds_value = response.get("idle_seconds")
    if isinstance(idle_seconds_value, (int, float)):
        freshness["idle_seconds"] = float(idle_seconds_value)
    window = response.get("window")
    if isinstance(window, dict):
        freshness["window"] = window
    if isinstance(freshness["capture_latency_seconds"], str):
        try:
            freshness["capture_latency_seconds"] = float(freshness["capture_latency_seconds"])
        except ValueError:
            freshness["capture_latency_seconds"] = None

    return image_data, freshness, time.monotonic()


def inspect_screen_activity(
    reason: str = "",
    detail_level: str = "auto",
    force_vision: bool = False,
    vision_max_requests_per_minute: int | None = None,
    provider: str = "auto",
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_model: str | None = None,
    gemini_api_key: str | None = None,
    gemini_api_keys: list[Any] | None = None,
    gemini_api_url: str | None = None,
    gemini_model: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Return a compact JSON summary of the user's current screen activity."""
    tool_started_at = datetime.now().isoformat(timespec="milliseconds")
    tool_started_monotonic = time.monotonic()
    detail_level = (detail_level or "auto").strip().lower()
    if detail_level not in {"auto", "brief", "normal", "visual"}:
        detail_level = "auto"
    if force_vision:
        detail_level = "visual"

    idle = idle_seconds()
    window = foreground_window_info()

    if idle >= DEFAULT_IDLE_THRESHOLD_SECONDS and not force_vision:
        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": "idle-detector",
            "model": "",
            "vision_rate_limited": False,
            "idle_seconds": round(idle, 2),
            "window": asdict(window),
            "activity": "idle",
            "intent": "用户已经一段时间没有操作电脑，当前不适合主动打扰。",
            "confidence": 0.95,
            "should_interrupt": False,
            "sensitive_visible": False,
            "summary": "用户当前处于挂机或离开状态。",
            "reason": reason,
        }
        return json.dumps(result, ensure_ascii=False)

    selected_provider = choose_provider(
        provider=provider,
        detail_level=detail_level,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        gemini_api_keys=gemini_api_keys,
    )

    vision_error = None
    vision_key_attempts: list[dict[str, Any]] = []
    vision_key_index = None
    freshness: dict[str, Any] | None = None
    capture_finished_monotonic: float | None = None
    if selected_provider == "heuristic":
        parsed = heuristic_activity(window)
        raw_text = json.dumps(parsed, ensure_ascii=False)
        model_name = "heuristic"
        vision_rate_limited = False
    else:
        limit = resolve_vision_limit(vision_max_requests_per_minute)
        vision_rate_limited = False
        raw_text = ""
        model_name = ""
        try:
            if selected_provider == "openai":
                if not force_vision:
                    allowed, retry_after = reserve_vision_request(limit)
                    if not allowed:
                        parsed = heuristic_activity(window)
                        raw_text = json.dumps(parsed, ensure_ascii=False)
                        result = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "provider": "heuristic-rate-limited",
                            "model": "heuristic",
                            "vision_rate_limited": True,
                            "vision_retry_after_seconds": retry_after,
                            "vision_max_requests_per_minute": limit,
                            "idle_seconds": round(idle, 2),
                            "window": asdict(window),
                            "activity": parsed["activity"],
                            "intent": parsed["intent"],
                            "confidence": parsed["confidence"],
                            "should_interrupt": parsed["should_interrupt"],
                            "sensitive_visible": parsed["sensitive_visible"],
                            "summary": parsed["summary"],
                            "reason": reason,
                            "raw_text": raw_text[:1000],
                        }
                        return json.dumps(result, ensure_ascii=False)

                model_name = openai_model or os.getenv("SCREEN_SKILL_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
                image_data, freshness, capture_finished_monotonic = capture_screen_for_vision(
                    provider=selected_provider,
                    model=model_name,
                )
                freshness["tool_started_at"] = tool_started_at
                client_window = freshness.get("window")
                if isinstance(client_window, dict):
                    window = WindowInfo(
                        title=str(client_window.get("title") or ""),
                        process_name=str(client_window.get("process_name") or ""),
                        process_path=str(client_window.get("process_path") or ""),
                    )
                if isinstance(freshness.get("idle_seconds"), (int, float)):
                    idle = float(freshness["idle_seconds"])
                prompt = build_prompt(window=window, idle=idle, reason=reason)
                mark_vision_model_start(freshness, provider=selected_provider, model=model_name, timeout_seconds=timeout_seconds)
                raw_text = analyze_with_openai(
                    image_data=image_data,
                    prompt=prompt,
                    api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com",
                    model=model_name,
                    timeout_seconds=timeout_seconds,
                )
                mark_vision_model_finish(freshness, provider=selected_provider, model=model_name, capture_finished_monotonic=capture_finished_monotonic)
            else:
                model_name = gemini_model or os.getenv("SCREEN_SKILL_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
                image_data, freshness, capture_finished_monotonic = capture_screen_for_vision(
                    provider=selected_provider,
                    model=model_name,
                )
                freshness["tool_started_at"] = tool_started_at
                client_window = freshness.get("window")
                if isinstance(client_window, dict):
                    window = WindowInfo(
                        title=str(client_window.get("title") or ""),
                        process_name=str(client_window.get("process_name") or ""),
                        process_path=str(client_window.get("process_path") or ""),
                    )
                if isinstance(freshness.get("idle_seconds"), (int, float)):
                    idle = float(freshness["idle_seconds"])
                prompt = build_prompt(window=window, idle=idle, reason=reason)
                mark_vision_model_start(freshness, provider=selected_provider, model=model_name, timeout_seconds=timeout_seconds)
                vision_result = analyze_with_gemini_key_pool(
                    image_data=image_data,
                    prompt=prompt,
                    api_key=gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                    api_keys=gemini_api_keys,
                    api_url=gemini_api_url,
                    model=model_name,
                    timeout_seconds=timeout_seconds,
                    max_requests_per_minute=limit,
                    bypass_rate_limit=force_vision,
                )
                raw_text = vision_result.text
                vision_key_index = vision_result.key_index
                vision_key_attempts = vision_result.attempts or []
                mark_vision_model_finish(freshness, provider=selected_provider, model=model_name, capture_finished_monotonic=capture_finished_monotonic)
            parsed = parse_model_json(raw_text)
        except VisionRateLimitError as error:
            parsed = heuristic_activity(window)
            raw_text = json.dumps(parsed, ensure_ascii=False)
            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "provider": "heuristic-rate-limited",
                "model": "heuristic",
                "vision_rate_limited": True,
                "vision_retry_after_seconds": error.retry_after_seconds,
                "vision_max_requests_per_minute": error.max_requests_per_minute,
                "idle_seconds": round(idle, 2),
                "window": asdict(window),
                "activity": parsed["activity"],
                "intent": parsed["intent"],
                "confidence": parsed["confidence"],
                "should_interrupt": parsed["should_interrupt"],
                "sensitive_visible": parsed["sensitive_visible"],
                "summary": parsed["summary"],
                "reason": reason,
                "raw_text": raw_text[:1000],
            }
            if error.attempts:
                result["vision_key_attempts"] = error.attempts
            if freshness:
                finalize_freshness(freshness, capture_finished_monotonic, tool_started_monotonic)
                result["freshness"] = freshness
            return json.dumps(result, ensure_ascii=False)
        except Exception as error:
            status = vision_error_status(error)
            vision_key_attempts = getattr(error, "attempts", None) or vision_key_attempts
            vision_error = {
                "type": type(error).__name__,
                "status": status,
            }
            if vision_key_attempts:
                vision_error["attempts"] = vision_key_attempts
            parsed = heuristic_activity(window)
            selected_provider = "heuristic-vision-error"
            vision_rate_limited = status == 429

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": selected_provider,
        "model": model_name,
        "vision_rate_limited": vision_rate_limited,
        "idle_seconds": round(idle, 2),
        "window": asdict(window),
        "activity": parsed["activity"],
        "intent": parsed["intent"],
        "confidence": parsed["confidence"],
        "should_interrupt": parsed["should_interrupt"],
        "sensitive_visible": parsed["sensitive_visible"],
        "summary": parsed["summary"],
        "reason": reason,
    }
    if vision_error:
        result["vision_error"] = vision_error
    if vision_key_index is not None:
        result["vision_key_index"] = vision_key_index
    if vision_key_attempts:
        result["vision_key_attempts"] = vision_key_attempts
    if freshness:
        finalize_freshness(freshness, capture_finished_monotonic, tool_started_monotonic)
        result["freshness"] = freshness
    return json.dumps(result, ensure_ascii=False)


def resolve_vision_limit(explicit_limit: int | None) -> int:
    raw_value = explicit_limit
    if raw_value is None:
        raw_value = os.getenv("SCREEN_SKILL_VISION_MAX_PER_MINUTE")
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_VISION_MAX_REQUESTS_PER_MINUTE


def reserve_vision_request(max_per_minute: int) -> tuple[bool, float]:
    now = time.monotonic()
    if max_per_minute <= 0:
        return False, 60.0

    with _vision_rate_lock:
        cutoff = now - 60.0
        while _vision_request_times and _vision_request_times[0] < cutoff:
            _vision_request_times.pop(0)

        if len(_vision_request_times) >= max_per_minute:
            retry_after = max(0.0, 60.0 - (now - _vision_request_times[0]))
            return False, round(retry_after, 2)

        _vision_request_times.append(now)
        return True, 0.0


def vision_error_status(error: Exception) -> int | None:
    status = getattr(error, "status", None)
    if status is not None:
        return status
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def normalize_gemini_key_candidates(
    api_key: str | None,
    api_keys: list[Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(key: Any, index: Any = None) -> None:
        key_text = str(key or "").strip()
        if not key_text or key_text in seen:
            return
        seen.add(key_text)
        try:
            normalized_index = int(index) if index is not None else None
        except (TypeError, ValueError):
            normalized_index = None
        candidates.append({"key": key_text, "index": normalized_index})

    for item in api_keys or []:
        if isinstance(item, dict):
            add_candidate(
                item.get("key") or item.get("api_key") or item.get("apiKey") or item.get("value"),
                item.get("index") if "index" in item else item.get("key_index"),
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            first, second = item[0], item[1]
            if isinstance(first, int):
                add_candidate(second, first)
            else:
                add_candidate(first, second if isinstance(second, int) else None)
        else:
            add_candidate(item)

    add_candidate(api_key)
    return candidates


def compact_response_error(response: requests.Response | None, max_message_chars: int = 220) -> str:
    if response is None:
        return ""

    try:
        data = response.json()
    except ValueError:
        text = (getattr(response, "text", "") or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text[:max_message_chars]

    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return json.dumps(data, ensure_ascii=False)[:max_message_chars]

    parts = []
    if error.get("code") is not None:
        parts.append(f"code={error.get('code')}")
    if error.get("status"):
        parts.append(f"status={error.get('status')}")

    quota_id = ""
    retry_delay = ""
    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        if not retry_delay and detail.get("retryDelay"):
            retry_delay = str(detail.get("retryDelay"))
        for violation in detail.get("violations") or []:
            if isinstance(violation, dict) and violation.get("quotaId"):
                quota_id = str(violation.get("quotaId"))
                break
        if quota_id and retry_delay:
            break

    if quota_id:
        parts.append(f"quota={quota_id}")
    if retry_delay:
        parts.append(f"retryDelay={retry_delay}")

    message = str(error.get("message") or "").strip()
    message = re.sub(r"\s+", " ", message)
    if message:
        parts.append(f"message={message[:max_message_chars]}")
    return "; ".join(parts)[: max_message_chars + 160]


def gemini_attempt_error_detail(error: Exception) -> str:
    response = getattr(error, "response", None)
    if response is not None:
        return compact_response_error(response, 260)
    if isinstance(error, requests.RequestException):
        return ""
    return str(error)[:500]


def attach_attempts(error: Exception, attempts: list[dict[str, Any]]) -> None:
    try:
        setattr(error, "attempts", attempts)
    except Exception:
        pass


def idle_seconds() -> float:
    if not _IS_WINDOWS:
        return 0.0
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
    return max(0.0, millis / 1000.0)


def foreground_window_info() -> WindowInfo:
    if not _IS_WINDOWS:
        return WindowInfo(title="", process_name="", process_path="")
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
    if not _IS_WINDOWS:
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
    if mss is None:
        raise RuntimeError("Server-side screen capture is unavailable. Start the desktop client screen source.")

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


def capture_screen_for_vision(
    provider: str,
    model: str,
) -> tuple[bytes, dict[str, Any], float]:
    client_capture = _request_client_screen_capture(
        {
            "provider": provider,
            "model": model,
            "max_width": DEFAULT_MAX_WIDTH,
            "jpeg_quality": DEFAULT_JPEG_QUALITY,
        }
    )
    if client_capture is not None:
        return _decode_client_capture_response(client_capture)
    if not DEFAULT_ALLOW_SERVER_CAPTURE:
        raise RuntimeError("No client screen source is available for this user.")

    capture_started_at = datetime.now().isoformat(timespec="milliseconds")
    capture_started_monotonic = time.monotonic()
    image = capture_screen(DEFAULT_MAX_WIDTH)
    image_data = jpeg_bytes(image, DEFAULT_JPEG_QUALITY)
    capture_finished_monotonic = time.monotonic()
    capture_finished_at = datetime.now().isoformat(timespec="milliseconds")
    capture_latency_seconds = round(capture_finished_monotonic - capture_started_monotonic, 3)
    freshness = {
        "capture_started_at": capture_started_at,
        "capture_finished_at": capture_finished_at,
        "capture_latency_seconds": capture_latency_seconds,
        "image_width": image.width,
        "image_height": image.height,
        "jpeg_bytes": len(image_data),
        "is_live_stream": False,
        "capture_source": "server",
    }
    return image_data, freshness, capture_finished_monotonic


def mark_vision_model_start(
    freshness: dict[str, Any],
    provider: str,
    model: str,
    timeout_seconds: float,
) -> None:
    freshness["vision_started_at"] = datetime.now().isoformat(timespec="milliseconds")
    freshness["_vision_started_monotonic"] = time.monotonic()


def mark_vision_model_finish(
    freshness: dict[str, Any],
    provider: str,
    model: str,
    capture_finished_monotonic: float | None,
) -> None:
    finished_monotonic = time.monotonic()
    freshness["vision_finished_at"] = datetime.now().isoformat(timespec="milliseconds")
    started_monotonic = freshness.get("_vision_started_monotonic")
    if isinstance(started_monotonic, (int, float)):
        freshness["vision_latency_seconds"] = round(finished_monotonic - started_monotonic, 3)
    if capture_finished_monotonic is not None:
        freshness["capture_age_at_vision_finish_seconds"] = round(
            finished_monotonic - capture_finished_monotonic,
            3,
        )


def finalize_freshness(
    freshness: dict[str, Any],
    capture_finished_monotonic: float | None,
    tool_started_monotonic: float,
) -> None:
    now = time.monotonic()
    freshness["returned_at"] = datetime.now().isoformat(timespec="milliseconds")
    freshness["tool_elapsed_seconds"] = round(now - tool_started_monotonic, 3)
    if capture_finished_monotonic is not None:
        freshness["capture_age_seconds"] = round(now - capture_finished_monotonic, 3)
    freshness.pop("_vision_started_monotonic", None)


def choose_provider(
    provider: str,
    detail_level: str,
    openai_api_key: str | None,
    gemini_api_key: str | None,
    gemini_api_keys: list[Any] | None = None,
) -> str:
    provider = (provider or "auto").strip().lower()
    configured = os.getenv("SCREEN_SKILL_PROVIDER", "").strip().lower()
    if provider == "auto" and configured:
        provider = configured

    if detail_level == "brief":
        return "heuristic"
    if provider in {"heuristic", "openai", "gemini"}:
        return provider
    if openai_api_key or os.getenv("OPENAI_API_KEY"):
        return "openai"
    if (
        gemini_api_key
        or normalize_gemini_key_candidates(gemini_api_key, gemini_api_keys)
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ):
        return "gemini"
    return "heuristic"


def build_prompt(window: WindowInfo, idle: float, reason: str) -> str:
    return (
        "你是桌宠的屏幕活动理解 skill。根据截图、前台窗口和空闲时间，判断用户当前可能在做什么。\n"
        "只输出 JSON，不要输出 Markdown。不要逐字转录私人聊天、密钥、账号、地址、代码中的密钥等敏感内容；"
        "如看见敏感内容，把 sensitive_visible 设为 true，但 summary 只做概括。\n"
        "should_interrupt 表示桌宠现在是否适合主动插话。\n"
        "JSON schema: {"
        "\"activity\":\"coding|reading|chatting|watching_video|gaming|writing|browsing|meeting|idle|unknown\","
        "\"intent\":\"一句话说明用户当前意图\","
        "\"confidence\":0.0,"
        "\"should_interrupt\":false,"
        "\"sensitive_visible\":false,"
        "\"summary\":\"一句话中文概括\""
        "}\n\n"
        f"触发原因: {reason or '模型主动请求查看当前屏幕状态'}\n"
        f"空闲秒数: {idle:.1f}\n"
        f"前台程序: {window.process_name or 'unknown'}\n"
        f"前台标题: {window.title or 'unknown'}"
    )


def analyze_with_openai(
    image_data: bytes,
    prompt: str,
    api_key: str | None,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured for screen skill.")
    endpoint = openai_responses_endpoint(base_url)
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
        "max_output_tokens": 350,
    }
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise
    data = response.json()
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def openai_responses_endpoint(base_url: str) -> str:
    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    if base_url.endswith("/responses"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/responses"
    if "api.openai.com" in base_url:
        return base_url + "/v1/responses"
    return base_url + "/responses"


def analyze_with_gemini_key_pool(
    image_data: bytes,
    prompt: str,
    api_key: str | None,
    api_keys: list[Any] | None,
    api_url: str | None,
    model: str,
    timeout_seconds: int,
    max_requests_per_minute: int,
    bypass_rate_limit: bool = False,
) -> VisionModelResult:
    candidates = normalize_gemini_key_candidates(api_key=api_key, api_keys=api_keys)
    if not candidates:
        raise RuntimeError("Gemini API key is not configured for screen skill.")

    attempts: list[dict[str, Any]] = []
    last_status: int | None = None

    for attempt_number, candidate in enumerate(candidates, start=1):
        if not bypass_rate_limit:
            allowed, retry_after = reserve_vision_request(max_requests_per_minute)
            if not allowed:
                raise VisionRateLimitError(
                    retry_after_seconds=retry_after,
                    max_requests_per_minute=max_requests_per_minute,
                    attempts=attempts,
                )

        key_index = candidate.get("index")
        attempt_started = time.monotonic()
        try:
            text = analyze_with_gemini(
                image_data=image_data,
                prompt=prompt,
                api_key=candidate["key"],
                api_url=api_url,
                model=model,
                timeout_seconds=timeout_seconds,
            )
            elapsed_seconds = round(time.monotonic() - attempt_started, 3)
            attempts.append({
                "attempt": attempt_number,
                "key_index": key_index,
                "status": 200,
                "error_type": "",
                "elapsed_seconds": elapsed_seconds,
            })
            return VisionModelResult(
                text=text,
                key_index=key_index,
                attempts=attempts,
            )
        except Exception as error:
            status = vision_error_status(error)
            last_status = status
            retryable = status in RETRYABLE_GEMINI_KEY_STATUSES
            detail = gemini_attempt_error_detail(error)
            attempt = {
                "attempt": attempt_number,
                "key_index": key_index,
                "status": status,
                "error_type": type(error).__name__,
                "retryable": retryable,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            }
            if detail:
                attempt["detail"] = detail
            attempts.append(attempt)
            if retryable:
                continue
            attach_attempts(error, attempts)
            raise

    raise GeminiKeyPoolExhaustedError(
        "All Gemini API keys failed for screen vision.",
        attempts=attempts,
        status=last_status,
    )


def analyze_with_gemini(
    image_data: bytes,
    prompt: str,
    api_key: str | None,
    api_url: str | None,
    model: str,
    timeout_seconds: int,
) -> str:
    if not api_key:
        raise RuntimeError("Gemini API key is not configured for screen skill.")
    endpoint = api_url or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        max_output_tokens = max(
            512,
            int(
                os.getenv(
                    "SCREEN_SKILL_GEMINI_MAX_OUTPUT_TOKENS",
                    str(DEFAULT_GEMINI_MAX_OUTPUT_TOKENS),
                )
            ),
        )
    except ValueError:
        max_output_tokens = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_data).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCREEN_RESPONSE_SCHEMA,
            "temperature": 0.1,
            "maxOutputTokens": max_output_tokens,
        },
    }
    try:
        response = requests.post(
            endpoint,
            params={"key": api_key},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise

    try:
        data = response.json()
    except ValueError as error:
        raise

    texts = []
    finish_reasons = []
    for candidate in data.get("candidates") or []:
        finish_reasons.append(candidate.get("finishReason") or "UNKNOWN")
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if text:
                texts.append(text)
    result = "\n".join(texts).strip()
    if not result:
        raise ValueError("Gemini screen vision response contains no text.")
    return result


def heuristic_activity(window: WindowInfo) -> dict[str, Any]:
    text = f"{window.process_name} {window.title}".lower()
    if any(value in text for value in ["code.exe", "pycharm", "visual studio", ".py", ".js", ".ts"]):
        activity = "coding"
        intent = "用户可能正在编写或调试代码。"
        confidence = 0.45
    elif any(value in text for value in ["chrome", "edge", "firefox", "browser"]):
        activity = "browsing"
        intent = "用户可能正在浏览网页。"
        confidence = 0.38
    elif any(value in text for value in ["wechat", "qq", "telegram", "discord"]):
        activity = "chatting"
        intent = "用户可能正在聊天。"
        confidence = 0.45
    elif any(value in text for value in ["steam", "game", "unity", "unreal"]):
        activity = "gaming"
        intent = "用户可能正在玩游戏。"
        confidence = 0.45
    else:
        activity = "unknown"
        intent = "仅凭窗口信息无法可靠判断用户意图。"
        confidence = 0.2

    return {
        "activity": activity,
        "intent": intent,
        "confidence": confidence,
        "should_interrupt": False,
        "sensitive_visible": False,
        "summary": intent,
    }


def parse_model_json(text: str) -> dict[str, Any]:
    text = (text or "").strip().lstrip("\ufeff")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        nested = data.get("result") or data.get("analysis")
        if isinstance(nested, dict):
            data = nested
    if not isinstance(data, dict):
        raise ValueError("Screen analysis result must be a JSON object.")

    activity = str(data.get("activity", "unknown")).strip().lower()
    if activity not in SCREEN_ANALYSIS_ACTIVITIES:
        activity = "unknown"
    return {
        "activity": activity,
        "intent": str(data.get("intent", "")),
        "confidence": clamp_float(data.get("confidence", 0.0)),
        "should_interrupt": parse_bool(
            data.get("should_interrupt", data.get("shouldInterrupt", False))
        ),
        "sensitive_visible": parse_bool(
            data.get("sensitive_visible", data.get("sensitiveVisible", False))
        ),
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
