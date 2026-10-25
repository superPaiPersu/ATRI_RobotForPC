import base64
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from urllib.parse import unquote, urlparse

import httpx
import requests
from google import genai
# from google.genai import types
import dashscope
import serpapi
import anthropic
from openai import OpenAI


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _compact_log_value(value, max_chars=180):
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _log_tool_call(name, arguments):
    print(f"tool call name={name} args={_compact_log_value(arguments)}")


def _log_model_reply(provider, content):
    text = _content_to_text(content)
    print(f"model reply provider={provider} chars={len(text)}")


def _is_summary_turn(messages):
    for message in _normalize_messages(messages):
        role = (message.get("role") or "user").lower()
        if role not in ("user",):
            continue

        content = (message.get("content") or "").strip()
        if content in (
                "[系统消息]:这是系统提示，对话即将结束。",
                "这是系统提示，对话即将结束。",
        ):
            return True
    return False


def _extract_json_object(text):
    text = _content_to_text(text).strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict):
        return payload
    return None


def _extract_field_by_regex(text, field_name, next_field_name=None):
    text = _content_to_text(text)
    if not text:
        return ""

    if next_field_name:
        pattern = rf'"{re.escape(field_name)}"\s*:\s*"(.*?)(?="\s*,\s*"{re.escape(next_field_name)}"\s*:)'
    else:
        pattern = rf'"{re.escape(field_name)}"\s*:\s*"(.*?)(?="\s*}})'

    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


def _extract_action_by_regex(text):
    text = _content_to_text(text)
    if not text:
        return ""

    match = re.search(
        r'"action"\s*:\s*"(reply|wait|history_require|summary|no_sense)"',
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    return ""


def _normalize_chat_payload(payload, raw_text=""):
    payload = payload if isinstance(payload, dict) else {}
    action = str(payload.get("action") or "").strip()
    raw_action = _extract_action_by_regex(raw_text)
    if action not in ("reply", "wait", "history_require"):
        action = raw_action

    if action in ("wait", "history_require"):
        return {"action": action}

    cn = str(payload.get("cn") or payload.get("content") or "").strip()
    jp = str(payload.get("jp") or "").strip()
    mood = str(payload.get("mood") or "").strip()
    expression = str(payload.get("expression") or "").strip()
    heart_beat_seconds = str(payload.get("heart-beat-seconds") or "").strip()
    special = str(payload.get("special") or "").strip()
    if not cn:
        cn = _extract_field_by_regex(raw_text, "cn", "jp")
    if not jp:
        jp = _extract_field_by_regex(raw_text, "jp", "mood")
    if not mood:
        mood = _extract_field_by_regex(raw_text, "mood", "expression")
    if not expression:
        expression = _extract_field_by_regex(raw_text, "expression", None)

    if not cn and not jp:
        fallback = raw_text.strip()
        if fallback.lower() in ("wait", "history_require"):
            return {"action": fallback.lower()}
        if fallback:
            cn = fallback
            jp = fallback
        else:
            return {"action": "wait"}

    if not cn:
        cn = jp or raw_text.strip() or "……"
    if not jp:
        jp = cn
    if not mood:
        mood = "normal"
    if not expression:
        expression = "normal"

    return {
        "action": "reply",
        "cn": cn,
        "jp": jp,
        "mood": mood,
        "expression": expression,
        "heart_beat_seconds": heart_beat_seconds,
        "special": special
    }


def _normalize_summary_payload(payload, raw_text=""):
    payload = payload if isinstance(payload, dict) else {}
    action = str(payload.get("action") or "").strip()
    raw_action = _extract_action_by_regex(raw_text)
    if action not in ("summary", "no_sense"):
        action = raw_action

    if action == "no_sense":
        return {"action": "no_sense"}

    user = str(payload.get("user") or "").strip()
    model = str(payload.get("model") or "").strip()
    if not user:
        user = _extract_field_by_regex(raw_text, "user", "model")
    if not model:
        model = _extract_field_by_regex(raw_text, "model", None)

    if not user and not model:
        return {"action": "no_sense"}

    if not user:
        user = model
    if not model:
        model = user

    return {
        "action": "summary",
        "user": user,
        "model": model,
    }


def _finalize_model_content(content, summary_mode=False):
    raw_text = _content_to_text(content).strip()
    payload = _extract_json_object(raw_text)
    if summary_mode:
        normalized = _normalize_summary_payload(payload, raw_text)
    else:
        normalized = _normalize_chat_payload(payload, raw_text)
    return json.dumps(normalized, ensure_ascii=False)


try:
    from server.screen_skill import inspect_screen_activity as _inspect_screen_activity_impl
except ImportError:
    from screen_skill import inspect_screen_activity as _inspect_screen_activity_impl

try:
    from server.file_skill import (
        list_authorized_dirs as _list_authorized_dirs,
        list_user_files as _list_user_files,
        read_user_file as _read_user_file,
        search_user_files as _search_user_files,
    )
except ImportError:
    from file_skill import (
        list_authorized_dirs as _list_authorized_dirs,
        list_user_files as _list_user_files,
        read_user_file as _read_user_file,
        search_user_files as _search_user_files,
    )

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
#
# grounding_tool = types.Tool(
#     google_search=types.GoogleSearch()
# )
#
# config = types.GenerateContentConfig(
#     tools=[grounding_tool]
# )

key_lt = [
    os.getenv("GEMINI_API_KEY", "")
]

key_index = 0
key_lock = threading.Lock()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_RESPONSES_URL = os.getenv("OPENAI_BASE_URL", "https://speed.ai-pixel.online/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "150"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://speed.ai-pixel.online/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

MAX_AGENT_STEPS = 4
_agent_step_limit_var: ContextVar[int | None] = ContextVar("agent_step_limit", default=None)
_disabled_tool_names_var: ContextVar[set[str] | None] = ContextVar("disabled_tool_names", default=None)
_request_api_url_var: ContextVar[str | None] = ContextVar("request_api_url", default=None)
_request_api_key_var: ContextVar[str | None] = ContextVar("request_api_key", default=None)
_request_model_name_var: ContextVar[str | None] = ContextVar("request_model_name", default=None)
MODEL_TIMEOUT_SECONDS = _env_float("GEMINI_TIMEOUT_SECONDS", 25)
SCREEN_SKILL_TIMEOUT_SECONDS = _env_float("SCREEN_SKILL_TIMEOUT_SECONDS", 12)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://api.yomiapi.com/v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_URL = f"{GEMINI_BASE_URL.rstrip('/')}/models/{GEMINI_MODEL}:generateContent"
BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"
RETRYABLE_GEMINI_KEY_STATUSES = {401, 403, 429}


def set_agent_step_limit(limit):
    return _agent_step_limit_var.set(limit)


def reset_agent_step_limit(token):
    _agent_step_limit_var.reset(token)


def set_disabled_tools(tool_names):
    return _disabled_tool_names_var.set(set(tool_names or []))


def reset_disabled_tools(token):
    _disabled_tool_names_var.reset(token)


def _set_request_endpoint(api_url, api_key):
    url_token = _request_api_url_var.set(str(api_url or "").strip())
    key_token = _request_api_key_var.set(str(api_key or "").strip())
    return url_token, key_token


def _reset_request_endpoint(tokens):
    url_token, key_token = tokens
    _request_api_key_var.reset(key_token)
    _request_api_url_var.reset(url_token)


def _current_request_api_url():
    return _request_api_url_var.get() or ""


def _current_request_api_key():
    return _request_api_key_var.get() or ""


def _set_request_model_name(model_name):
    return _request_model_name_var.set(str(model_name or "").strip().lower())


def _reset_request_model_name(token):
    _request_model_name_var.reset(token)


def _current_request_model_name():
    return _request_model_name_var.get() or ""


def _gemini_api_url(base_url, model):
    base_url = str(base_url or "").strip()
    if ":generateContent" in base_url:
        return base_url
    return f"{base_url.rstrip('/')}/models/{model}:generateContent"


def _gemini_generation_config(model):
    config = {
        "responseMimeType": "application/json",
    }
    model_name = str(model or "").lower()
    if model_name.startswith("gemini-3"):
        config["thinkingConfig"] = {
            "thinkingLevel": "HIGH",
            "includeThoughts": False,
        }
    elif "2.5" in model_name:
        config["thinkingConfig"] = {
            "thinkingBudget": -1,
            "includeThoughts": False,
        }
    return config


def _agent_step_limit():
    value = _agent_step_limit_var.get()
    try:
        if value is not None:
            return max(1, min(int(value), 12))
    except (TypeError, ValueError):
        pass
    return MAX_AGENT_STEPS


def _tool_enabled(name):
    disabled = _disabled_tool_names_var.get() or set()
    return name not in disabled


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    texts.append(str(text))
        return "\n".join(texts)
    return str(content)


def _iter_content_items(content):
    if content is None:
        return []
    if isinstance(content, list):
        return content
    if isinstance(content, tuple):
        return list(content)
    if isinstance(content, dict):
        return [content]
    return [content]


def _extract_text_from_content_item(item):
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()

    item_type = str(item.get("type") or "").lower()
    if item_type in ("text", "input_text") or "text" in item or "content" in item:
        text = item.get("text") or item.get("content")
        return str(text or "").strip()
    return ""


def _extract_image_reference_from_content_item(item):
    if not isinstance(item, dict):
        return ""

    item_type = str(item.get("type") or "").lower()
    if item_type not in ("image_url", "input_image", "image") and not any(
            key in item for key in ("image_url", "url", "data", "file_uri")
    ):
        return ""

    image_ref = item.get("image_url") or item.get("url") or item.get("data") or item.get("file_uri")
    if isinstance(image_ref, dict):
        image_ref = image_ref.get("url") or image_ref.get("data") or image_ref.get("file_uri")
    return str(image_ref or "").strip()


def _guess_image_mime_type(image_bytes, fallback="image/png", path=""):
    path = str(path or "").strip()
    if path:
        guessed_type, _ = mimetypes.guess_type(path)
        if guessed_type:
            return guessed_type

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes))
        format_name = str(image.format or "").upper()
    except Exception:
        format_name = ""

    format_map = {
        "PNG": "image/png",
        "JPG": "image/jpeg",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
        "ICO": "image/x-icon",
    }
    return format_map.get(format_name, fallback)


def _image_reference_to_bytes(image_ref):
    image_ref = str(image_ref or "").strip()
    if not image_ref:
        return None, ""

    if image_ref.startswith("data:"):
        header, sep, encoded = image_ref.partition(",")
        if not sep or ";base64" not in header.lower():
            raise ValueError("Only base64 data URLs are supported for images.")
        mime_type = header[5:].split(";", 1)[0] or "image/png"
        return base64.b64decode(encoded), mime_type

    if re.match(r"^https?://", image_ref, flags=re.IGNORECASE):
        response = requests.get(image_ref, timeout=20)
        response.raise_for_status()
        mime_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if not mime_type:
            mime_type = _guess_image_mime_type(response.content)
        return response.content, mime_type or "image/png"

    local_path = image_ref
    if image_ref.startswith("file://"):
        parsed = urlparse(image_ref)
        local_path = unquote(parsed.path or "")
        if re.match(r"^/[A-Za-z]:", local_path):
            local_path = local_path.lstrip("/")

    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            image_bytes = f.read()
        mime_type = _guess_image_mime_type(image_bytes, path=local_path)
        return image_bytes, mime_type

    return None, ""


def _image_reference_to_data_url(image_ref):
    image_bytes, mime_type = _image_reference_to_bytes(image_ref)
    if not image_bytes:
        return None, "", "No image bytes found."
    mime_type = mime_type or "image/png"
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return data_url, mime_type, ""


def _image_reference_to_gemini_inline_data(image_ref):
    image_bytes, mime_type = _image_reference_to_bytes(image_ref)
    if not image_bytes:
        return None, "", "No image bytes found."
    mime_type = mime_type or "image/png"
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    }, mime_type, ""


def _ocr_text_for_content_item(item):
    image_ref = _extract_image_reference_from_content_item(item)
    if not image_ref:
        return ""
    ocr_text, error = _ocr_image_reference(image_ref)
    if ocr_text:
        return "【图片 OCR 结果】\n" + ocr_text
    return "【图片 OCR 失败】" + error


def _content_to_ocr_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content).strip()

    texts = []
    for item in content:
        text = _extract_text_from_content_item(item)
        if text:
            texts.append(text)
            continue

        image_text = _ocr_text_for_content_item(item)
        if image_text:
            texts.append(image_text)

    return "\n".join(texts).strip()


def _model_supports_vision(provider, model):
    model_name = str(model or "").lower()
    if provider == "gemini":
        return model_name.startswith("gemini-") or model_name.startswith("gemini")
    if provider == "gpt":
        return bool(re.match(r"^(gpt-(4o|4\.1|4\.5|5)([.-].*)?|o[134]([.-].*)?)$", model_name))
    if provider == "deepseek":
        return any(token in model_name for token in ("vision", "-vl", "vl-", "multimodal"))
    return False


def _current_screen_vision_provider_and_model():
    model_key = _current_request_model_name()
    if model_key == "gpt" and _model_supports_vision("gpt", OPENAI_MODEL):
        return "openai", OPENAI_MODEL
    if model_key == "gemini" and _model_supports_vision("gemini", GEMINI_MODEL):
        return "gemini", GEMINI_MODEL
    return "", ""


def _screen_vision_fallback_provider(primary_provider):
    primary_provider = str(primary_provider or "").strip().lower()
    if primary_provider == "openai":
        if _get_gemini_api_key_candidates() or GEMINI_API_KEY:
            return "gemini", GEMINI_MODEL
    elif primary_provider == "gemini":
        if OPENAI_API_KEY:
            return "openai", os.getenv("SCREEN_SKILL_OPENAI_MODEL", OPENAI_MODEL)
    return "", ""


def _screen_result_status(result):
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    vision_error = data.get("vision_error")
    if not isinstance(vision_error, dict):
        return None
    try:
        return int(vision_error.get("status"))
    except (TypeError, ValueError):
        return None


def _iter_messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    normalized = []
    for message in messages or []:
        if isinstance(message, str):
            normalized.append({"role": "user", "content": message})
            continue

        if not isinstance(message, dict):
            continue

        role = (message.get("role") or "user").lower()
        content = message.get("content")
        if content is None:
            content = message.get("parts")
        if content is None:
            content = message.get("text")

        normalized.append({"role": role, "content": content})

    return normalized


def _serialize_content_for_model(content, provider, vision_enabled):
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content).strip()

    text_chunks = []
    parts = []
    has_native_image = False

    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                text_chunks.append(text)
                if vision_enabled:
                    if provider == "gemini":
                        parts.append({"text": text})
                    elif provider == "gpt":
                        parts.append({"type": "input_text", "text": text})
                    else:
                        parts.append({"type": "text", "text": text})
            continue

        if not isinstance(item, dict):
            text = str(item).strip()
            if text:
                text_chunks.append(text)
                if vision_enabled:
                    if provider == "gemini":
                        parts.append({"text": text})
                    elif provider == "gpt":
                        parts.append({"type": "input_text", "text": text})
                    else:
                        parts.append({"type": "text", "text": text})
            continue

        text = _extract_text_from_content_item(item)
        if text:
            text_chunks.append(text)
            if vision_enabled:
                if provider == "gemini":
                    parts.append({"text": text})
                elif provider == "gpt":
                    parts.append({"type": "input_text", "text": text})
                else:
                    parts.append({"type": "text", "text": text})

        image_ref = _extract_image_reference_from_content_item(item)
        if not image_ref:
            continue

        if vision_enabled:
            if provider == "gemini":
                image_part, _, _ = _image_reference_to_gemini_inline_data(image_ref)
            else:
                data_url, _, _ = _image_reference_to_data_url(image_ref)
                image_part = None
                if data_url:
                    if provider == "gpt":
                        image_part = {"type": "input_image", "image_url": data_url}
                    elif provider == "deepseek":
                        image_part = {"type": "image_url", "image_url": {"url": data_url}}
                    else:
                        image_part = {"type": "image_url", "image_url": {"url": data_url}}

            if image_part:
                parts.append(image_part)
                has_native_image = True
                continue

        image_text = _ocr_text_for_content_item(item)
        if image_text:
            text_chunks.append(image_text)

    if vision_enabled and has_native_image:
        return parts

    return "\n".join(text_chunks).strip()


def _extract_image_url_from_part(item):
    if not isinstance(item, dict):
        return ""

    item_type = item.get("type")
    if item_type not in ("image_url", "input_image") and "image_url" not in item and "url" not in item:
        return ""

    image_url = item.get("image_url") or item.get("url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return str(image_url or "").strip()


def _image_url_to_bytes(url):
    return _image_reference_to_bytes(url)


def _ocr_with_rapidocr(image_bytes):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:
        return None, f"rapidocr unavailable: {type(e).__name__}: {e}"

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        engine = RapidOCR()
        result, _ = engine(image)
        lines = []
        for item in result or []:
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]))
        return "\n".join(lines).strip(), ""
    except Exception as e:
        return None, f"rapidocr failed: {type(e).__name__}: {e}"


def _ocr_with_pytesseract(image_bytes):
    try:
        import pytesseract
        from PIL import Image
    except Exception as e:
        return None, f"pytesseract unavailable: {type(e).__name__}: {e}"

    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=os.getenv("AI_PET_OCR_LANG", "chi_sim+eng"))
        return text.strip(), ""
    except Exception as e:
        return None, f"pytesseract failed: {type(e).__name__}: {e}"


def _ocr_with_tesseract_cli(image_bytes):
    executable = shutil.which(os.getenv("TESSERACT_CMD", "tesseract"))
    if not executable:
        return None, "tesseract cli unavailable"

    suffix = ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
        image_file.write(image_bytes)
        image_path = image_file.name

    try:
        result = subprocess.run(
            [executable, image_path, "stdout", "-l", os.getenv("AI_PET_OCR_LANG", "chi_sim+eng")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return None, (result.stderr or result.stdout or "tesseract failed").strip()
        return result.stdout.strip(), ""
    except Exception as e:
        return None, f"tesseract failed: {type(e).__name__}: {e}"
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass


def _ocr_image_url(url):
    try:
        image_bytes, mime_type = _image_url_to_bytes(url)
    except Exception as e:
        return "", f"Cannot decode image for OCR: {type(e).__name__}: {e}"

    if not image_bytes:
        return "", "No image bytes found for OCR."

    errors = []
    for runner in (_ocr_with_rapidocr, _ocr_with_pytesseract, _ocr_with_tesseract_cli):
        text, error = runner(image_bytes)
        if text:
            return text, ""
        if error:
            errors.append(error)

    return "", "OCR engine unavailable or returned no text. " + " | ".join(errors)


def _ocr_image_reference(image_ref):
    return _ocr_image_url(image_ref)


def _normalize_deepseek_content(content):
    return _content_to_ocr_text(content)


def _normalize_messages(messages, process_image_ocr=False):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    normalized = []
    for message in messages or []:
        if isinstance(message, str):
            normalized.append({"role": "user", "content": message})
            continue

        if not isinstance(message, dict):
            continue

        role = (message.get("role") or "user").lower()
        content = message.get("content")
        if content is None:
            content = message.get("parts")
        if content is None:
            content = message.get("text")

        if process_image_ocr:
            content = _normalize_deepseek_content(content)
            if isinstance(content, str):
                content = content.strip()
        else:
            content = _content_to_text(content).strip()
        if not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def _messages_to_openai_input(messages):
    vision_enabled = _model_supports_vision("gpt", OPENAI_MODEL)
    conversation = []

    for message in _iter_messages(messages):
        role = (message.get("role") or "user").lower()
        content = message.get("content")

        if role in ("system", "developer"):
            text = _content_to_ocr_text(content).strip()
            if text:
                conversation.append({"role": role, "content": text})
            continue

        if role in ("assistant", "model"):
            text = _content_to_ocr_text(content).strip()
            if text:
                conversation.append({"role": "assistant", "content": text})
            continue

        serialized = _serialize_content_for_model(content, "gpt", vision_enabled)
        if serialized:
            conversation.append({"role": "user", "content": serialized})

    return conversation


def _messages_to_deepseek_messages(messages):
    vision_enabled = _model_supports_vision("deepseek", DEEPSEEK_MODEL)
    system_parts = []
    chat_messages = []
    seen_chat_message = False

    for message in _iter_messages(messages):
        role = (message.get("role") or "user").lower()
        content = message.get("content")
        if not content:
            continue

        if role in ("system", "developer"):
            content_text = _content_to_ocr_text(content).strip()
            if not content_text:
                continue
            if seen_chat_message:
                chat_messages.append({
                    "role": "user",
                    "content": "【系统上下文】" + content_text,
                })
            else:
                system_parts.append(content_text)
            continue

        if role in ("assistant", "model"):
            serialized = _content_to_ocr_text(content).strip()
            if not serialized:
                continue
            seen_chat_message = True
            chat_messages.append({"role": "assistant", "content": serialized})
            continue

        serialized = _serialize_content_for_model(content, "deepseek", vision_enabled)
        if not serialized:
            continue

        seen_chat_message = True
        chat_messages.append({"role": "user", "content": serialized})

    if system_parts:
        chat_messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    return chat_messages


def _extract_response_text(data):
    if isinstance(data, dict):
        if data.get("output_text"):
            return data["output_text"]

        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            return _content_to_text(content)

        output = data.get("output") or []
    elif isinstance(data, list):
        output = data
    else:
        return ""

    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue

        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text"):
                return part.get("text") or ""

    return ""


def _get_current_time(timezone_name="Asia/Shanghai"):
    fallback_zones = {
        "Asia/Shanghai": timezone(timedelta(hours=8), "CST"),
        "UTC": timezone.utc,
    }

    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = fallback_zones.get(timezone_name)

    if tz is None:
        return json.dumps(
            {"error": f"Unknown timezone: {timezone_name}"},
            ensure_ascii=False,
        )

    now = datetime.now(tz)
    return json.dumps(
        {
            "timezone": timezone_name,
            "datetime": now.isoformat(timespec="seconds"),
            "weekday": now.strftime("%A"),
        },
        ensure_ascii=False,
    )


def _bocha_web_search(query, freshness="noLimit", summary=True, count=5):
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    if not api_key:
        return json.dumps(
            {"error": "BOCHA_API_KEY is not configured on the server."},
            ensure_ascii=False,
        )

    try:
        count = max(1, min(int(count), 10))
    except (TypeError, ValueError):
        count = 5
    if isinstance(summary, str):
        summary = summary.strip().lower() not in ("false", "0", "no", "off")

    payload = {
        "query": str(query or "").strip(),
        "freshness": freshness or "noLimit",
        "summary": bool(summary),
        "count": count,
    }
    if not payload["query"]:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    try:
        response = requests.post(
            BOCHA_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = e.response.text[:500]
        return json.dumps(
            {"error": f"Bocha web search failed: {e}", "detail": detail},
            ensure_ascii=False,
        )
    except ValueError as e:
        return json.dumps(
            {"error": f"Bocha response is not valid JSON: {e}"},
            ensure_ascii=False,
        )

    web_pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    results = []
    for item in web_pages[:count]:
        if not isinstance(item, dict):
            continue
        results.append({
            "name": item.get("name") or item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": item.get("snippet") or item.get("summary") or "",
            "site_name": item.get("siteName") or item.get("site") or "",
            "date_published": item.get("datePublished") or item.get("dateLastCrawled") or "",
        })

    return json.dumps(
        {
            "query": payload["query"],
            "results": results,
            "raw_answer": (data.get("data") or {}).get("answer") or "",
        },
        ensure_ascii=False,
    )


def _inspect_screen_activity(reason="", detail_level="auto", force_vision=False):
    if isinstance(force_vision, str):
        force_vision = force_vision.strip().lower() in ("1", "true", "yes", "on")

    screen_vision_provider, screen_vision_model = _current_screen_vision_provider_and_model()
    force_native_vision = bool(screen_vision_provider)
    if force_native_vision:
        force_vision = True
        detail_level = "visual"

    gemini_key_candidates = _get_gemini_api_key_candidates()
    current_gemini_key = gemini_key_candidates[0]["key"] if gemini_key_candidates else None
    request_api_url = _current_request_api_url()
    request_api_key = _current_request_api_key()

    def call_screen_vision(provider, model, use_request_endpoint):
        if provider == "openai":
            openai_key = request_api_key if use_request_endpoint else OPENAI_API_KEY
            openai_url = request_api_url if use_request_endpoint else OPENAI_RESPONSES_URL
            gemini_key = current_gemini_key
            gemini_keys = gemini_key_candidates
            gemini_url = _gemini_api_url(GEMINI_BASE_URL, GEMINI_MODEL)
        elif provider == "gemini":
            openai_key = OPENAI_API_KEY
            openai_url = OPENAI_RESPONSES_URL
            gemini_key = (request_api_key if use_request_endpoint else current_gemini_key) or GEMINI_API_KEY
            gemini_keys = [] if use_request_endpoint and request_api_key else gemini_key_candidates
            gemini_url = (
                _gemini_api_url(request_api_url, GEMINI_MODEL)
                if use_request_endpoint and request_api_url
                else _gemini_api_url(GEMINI_BASE_URL, GEMINI_MODEL)
            )
        else:
            openai_key = request_api_key or OPENAI_API_KEY
            openai_url = request_api_url or OPENAI_RESPONSES_URL
            gemini_key = request_api_key or current_gemini_key or GEMINI_API_KEY
            gemini_keys = [] if request_api_key else gemini_key_candidates
            gemini_url = _gemini_api_url(request_api_url, GEMINI_MODEL) if request_api_url else ""

        return _inspect_screen_activity_impl(
            reason=reason,
            detail_level=detail_level,
            force_vision=bool(force_vision),
            provider=provider or os.getenv("SCREEN_SKILL_PROVIDER", "openai"),
            openai_api_key=openai_key,
            openai_base_url=openai_url,
            openai_model=model if provider == "openai" and model else os.getenv("SCREEN_SKILL_OPENAI_MODEL",
                                                                                OPENAI_MODEL),
            gemini_api_key=gemini_key,
            gemini_api_keys=gemini_keys,
            gemini_api_url=gemini_url,
            gemini_model=model if provider == "gemini" and model else GEMINI_MODEL,
            timeout_seconds=SCREEN_SKILL_TIMEOUT_SECONDS,
        )

    primary_provider = screen_vision_provider or os.getenv("SCREEN_SKILL_PROVIDER", "openai")
    primary_model = screen_vision_model
    result = call_screen_vision(primary_provider, primary_model, use_request_endpoint=bool(screen_vision_provider))
    if force_native_vision and _screen_result_status(result) in {502, 503, 504}:
        fallback_provider, fallback_model = _screen_vision_fallback_provider(primary_provider)
        if fallback_provider:
            fallback_result = call_screen_vision(fallback_provider, fallback_model, use_request_endpoint=False)
            if _screen_result_status(fallback_result) not in {502, 503, 504}:
                result = fallback_result
    try:
        result_data = json.loads(result)
        vision_key_index = result_data.get("vision_key_index")
        if vision_key_index is not None:
            _set_gemini_api_key_index(vision_key_index)
    except (AttributeError, TypeError, ValueError):
        pass
    return result


LOCAL_TOOLS = {
    "get_current_time": _get_current_time,
    "bocha_web_search": _bocha_web_search,
    "inspect_screen_activity": _inspect_screen_activity,
    "list_authorized_dirs": _list_authorized_dirs,
    "list_user_files": _list_user_files,
    "search_user_files": _search_user_files,
    "read_user_file": _read_user_file,
}

LOCAL_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current date and time for an IANA timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone_name": {
                    "type": "string",
                    "description": "IANA timezone name, for example Asia/Shanghai or UTC.",
                }
            },
            "required": ["timezone_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "bocha_web_search",
        "description": "Search the web through Bocha AI and return current web results with URLs and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "freshness": {
                    "type": "string",
                    "description": "Freshness filter: noLimit, oneDay, oneWeek, oneMonth, or oneYear.",
                    "enum": ["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"],
                },
                "summary": {
                    "type": "boolean",
                    "description": "Whether Bocha should include generated summaries when available.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return, from 1 to 10.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "inspect_screen_activity",
        "description": (
            "偷偷看用户在干什么: inspect the user's current local screen state and return a compact "
            "JSON summary of activity, likely intent, foreground window, idle state, confidence, "
            "and whether the assistant should interrupt. The assistant may freely choose this tool "
            "before any reply when screen context would help. When the active chat model supports "
            "vision, the server forces fresh screenshot vision analysis on every call. If the result "
            "includes a vision_error, use the activity/summary fields instead of exposing transport "
            "status codes to the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief reason for looking at the screen.",
                },
                "detail_level": {
                    "type": "string",
                    "description": "visual requests screenshot analysis; auto lets the server choose.",
                    "enum": ["auto", "brief", "normal", "visual"],
                },
                "force_vision": {
                    "type": "boolean",
                    "description": "Whether to request fresh visual screenshot analysis.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_authorized_dirs",
        "description": (
            "List the local directories that the desktop client has authorized for file access. "
            "Use this before searching or reading local user files when the relevant directory is unknown."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_user_files",
        "description": (
            "List text/code files under an authorized local directory. Use this to browse a project "
            "or folder before deciding which specific file to read."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Authorized directory to list. Leave empty to list from all authorized roots.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Recursive depth from 0 to 8.",
                    "minimum": 0,
                    "maximum": 8,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to return.",
                    "minimum": 1,
                    "maximum": 500,
                },
                "extensions": {
                    "type": "array",
                    "description": "Optional allowed extensions such as .py, .md, .json.",
                    "items": {"type": "string"},
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_user_files",
        "description": (
            "Search authorized local text/code files by filename, path, and small content snippets. "
            "Use this when the user asks about a project or document but does not give an exact path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms, such as a filename, symbol, feature, or error text.",
                },
                "directory": {
                    "type": "string",
                    "description": "Optional authorized directory to narrow the search.",
                },
                "extensions": {
                    "type": "array",
                    "description": "Optional allowed extensions such as .py, .md, .json.",
                    "items": {"type": "string"},
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                    "minimum": 1,
                    "maximum": 100,
                },
                "max_files_scanned": {
                    "type": "integer",
                    "description": "Maximum number of files to scan before stopping.",
                    "minimum": 100,
                    "maximum": 20000,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_user_file",
        "description": (
            "Read a local text, code, PDF, or .docx Word file from an authorized directory. "
            "Use search_user_files or list_user_files first unless the user gave an exact authorized path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path, or a path relative to an authorized directory.",
                },
                "start": {
                    "type": "integer",
                    "description": "Character offset for reading a later chunk.",
                    "minimum": 0,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return, from 1000 to 50000.",
                    "minimum": 1000,
                    "maximum": 50000,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    }
]

DEEPSEEK_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }
    for tool in LOCAL_TOOL_DEFINITIONS
]


def _active_local_tool_definitions():
    return [
        tool for tool in LOCAL_TOOL_DEFINITIONS
        if _tool_enabled(tool.get("name"))
    ]


def _active_deepseek_tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in _active_local_tool_definitions()
    ]


def _gemini_schema(schema):
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        if key in {"additionalProperties", "strict", "minimum", "maximum"}:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                name: _gemini_schema(prop)
                for name, prop in value.items()
                if isinstance(prop, dict)
            }
        elif key == "items":
            cleaned[key] = _gemini_schema(value)
        elif isinstance(value, dict):
            cleaned[key] = _gemini_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _gemini_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _gemini_tool_declarations(tool_definitions=None):
    declarations = []
    selected_tools = LOCAL_TOOL_DEFINITIONS if tool_definitions is None else tool_definitions
    for tool in selected_tools:
        parameters = _gemini_schema(tool.get("parameters") or {"type": "object", "properties": {}})
        declarations.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": parameters,
        })
    return [{"functionDeclarations": declarations}]


def _active_gemini_tool_declarations():
    return _gemini_tool_declarations(_active_local_tool_definitions())


def _active_gemini_function_names():
    return [
        tool["name"]
        for tool in _active_local_tool_definitions()
    ]


GEMINI_TOOL_DECLARATIONS = _gemini_tool_declarations()


def _extract_function_calls(data):
    output = data if isinstance(data, list) else data.get("output", [])
    return [
        item for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _run_local_tool(function_call):
    name = function_call.get("name")
    raw_arguments = function_call.get("arguments") or "{}"

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid tool arguments: {e}"}, ensure_ascii=False)

    return _run_local_tool_by_name(name, arguments)


def _run_local_tool_by_name(name, arguments):
    _log_tool_call(name, arguments)
    if not _tool_enabled(name):
        return json.dumps({"error": f"Tool is unavailable for this client: {name}"}, ensure_ascii=False)
    tool = LOCAL_TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)

    try:
        return tool(**arguments)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _run_chat_completion_tool_call(tool_call):
    function = getattr(tool_call, "function", None)
    if function is None and isinstance(tool_call, dict):
        function = tool_call.get("function") or {}

    name = getattr(function, "name", None)
    if name is None and isinstance(function, dict):
        name = function.get("name")

    raw_arguments = getattr(function, "arguments", None)
    if raw_arguments is None and isinstance(function, dict):
        raw_arguments = function.get("arguments")
    raw_arguments = raw_arguments or "{}"

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid tool arguments: {e}"}, ensure_ascii=False)

    return _run_local_tool_by_name(name, arguments)


def _chat_tool_call_to_message(tool_call):
    function = getattr(tool_call, "function", None)
    if function is None and isinstance(tool_call, dict):
        function = tool_call.get("function") or {}

    return {
        "id": getattr(tool_call, "id", None) if not isinstance(tool_call, dict) else tool_call.get("id"),
        "type": (getattr(tool_call, "type", None) if not isinstance(tool_call, dict) else tool_call.get(
            "type")) or "function",
        "function": {
            "name": getattr(function, "name", None) if not isinstance(function, dict) else function.get("name"),
            "arguments": getattr(function, "arguments", None) if not isinstance(function, dict) else function.get(
                "arguments", "{}"),
        },
    }


def _require_request_endpoint(api_url, api_key, provider):
    api_url = str(api_url or "").strip()
    api_key = str(api_key or "").strip()
    if not api_url:
        raise ValueError(f"{provider} api_url is required")
    if not api_key:
        raise ValueError(f"{provider} api_key is required")
    return api_url, api_key


def _create_openai_client(api_url, api_key):
    api_url, api_key = _require_request_endpoint(api_url, api_key, "GPT")
    timeout = httpx.Timeout(
        timeout=OPENAI_TIMEOUT_SECONDS,
        connect=10.0,
        read=OPENAI_TIMEOUT_SECONDS,
        write=20.0,
        pool=10.0,
    )
    return OpenAI(
        api_key=api_key,
        base_url=api_url,
        timeout=timeout,
        max_retries=OPENAI_MAX_RETRIES,
    )


def _create_gpt_response(client, conversation_input, tools, include_web_search):
    request = {
        "model": OPENAI_MODEL,
        "input": conversation_input,
        "tools": tools,
        "tool_choice": "auto",
    }
    if include_web_search:
        request["include"] = ["web_search_call.action.sources"]
    return client.responses.create(**request)


def _is_timeout_error(error):
    return type(error).__name__ in {
        "APITimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
    }


def _model_error_message(error):
    return {
        "role": "assistant",
        "content": f"妯″瀷璇锋眰瓒呮椂鎴栬繛鎺ヤ腑鏂紝璇风◢鍚庨噸璇曘€傞敊璇細{type(error).__name__}: {error}",
    }


def _json_reply_message(cn, jp=None, mood="normal", expression="normal"):
    content = json.dumps(
        {
            "action": "reply",
            "cn": cn,
            "jp": jp or cn,
            "mood": mood,
            "expression": expression,
        },
        ensure_ascii=False,
    )
    _log_model_reply("server", content)
    return {
        "role": "assistant",
        "content": content,
    }


def _fallback_from_tool_outputs(tool_outputs, error=None):
    screen_result = None
    for output in reversed(tool_outputs or []):
        raw_output = output.get("output") if isinstance(output, dict) else None
        if not raw_output:
            continue
        try:
            data = json.loads(raw_output)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("activity"):
            screen_result = data
            break

    if not screen_result:
        return None

    window = screen_result.get("window") or {}
    app_name = window.get("process_name") or "当前窗口"
    intent = screen_result.get("intent") or screen_result.get("summary") or "我看到了你当前的屏幕状态。"
    confidence = screen_result.get("confidence")
    suffix = ""
    if isinstance(confidence, (int, float)):
        suffix = f"（置信度 {confidence:.2f}）"
    if screen_result.get("vision_rate_limited"):
        suffix += " 视觉分析额度暂时用完了，所以这次只用了本地窗口分析。"
    freshness = screen_result.get("freshness") or {}
    capture_age = freshness.get("capture_age_seconds")
    if isinstance(capture_age, (int, float)) and capture_age >= 2:
        suffix += f" 这张截图到回复时已经过去约 {capture_age:.1f} 秒。"

    cn = f"我刚刚偷偷看了一眼，你现在大概在 {app_name} 里：{intent}{suffix}"
    return _json_reply_message(cn, cn, mood="normal", expression="normal")


def get_answer_gpt(api_url, api_key, messages):
    conversation_input = _messages_to_openai_input(messages)
    summary_mode = _is_summary_turn(conversation_input)

    try:
        client = _create_openai_client(api_url, api_key)
    except ValueError as e:
        return _json_reply_message(str(e), str(e))

    local_tools = _active_local_tool_definitions()
    tools = [{"type": "web_search"}, *local_tools]
    last_tool_outputs = []

    for step in range(_agent_step_limit()):
        try:
            response = _create_gpt_response(
                client,
                conversation_input,
                tools,
                include_web_search=True,
            )
        except Exception as e:
            if step == 0 and _is_timeout_error(e):
                try:
                    response = _create_gpt_response(
                        client,
                        conversation_input,
                        local_tools,
                        include_web_search=False,
                    )
                except Exception as retry_error:
                    return _json_reply_message(
                        f"GPT 中转请求超时，请稍后重试。错误：{type(retry_error).__name__}: {retry_error}",
                        "GPTの中継リクエストがタイムアウトしました。少し待ってからもう一度試してください。",
                    )
            else:
                fallback = _fallback_from_tool_outputs(last_tool_outputs, e)
                if fallback:
                    return fallback
                return _json_reply_message(
                    f"模型请求失败，请稍后重试。错误：{type(e).__name__}: {e}",
                    "モデルのリクエストに失敗しました。少し待ってからもう一度試してください。",
                )

        data = response.model_dump()
        function_calls = _extract_function_calls(data)

        if not function_calls:
            content = _extract_response_text(data)
            if content:
                final_content = _finalize_model_content(content, summary_mode)
                _log_model_reply("gpt", final_content)
                return {"role": "assistant", "content": final_content}
            return {
                "role": "assistant",
                "content": _finalize_model_content("", summary_mode),
            }

        output_items = data.get("output", [])
        tool_outputs = [
            {
                "type": "function_call_output",
                "call_id": function_call.get("call_id"),
                "output": _run_local_tool(function_call),
            }
            for function_call in function_calls
        ]
        last_tool_outputs = tool_outputs
        conversation_input = [*conversation_input, *output_items, *tool_outputs]

    return _json_reply_message(
        "搜索步骤过多，我先停一下；你可以换个更具体的问题再试。",
        "検索の手順が多すぎたので、いったん止めます。もう少し具体的に聞いてください。",
    )


def get_ans_deepseek(api_url, api_key, messages):
    conversation_input = _messages_to_deepseek_messages(messages)
    summary_mode = _is_summary_turn(conversation_input)
    try:
        request_api_url, request_api_key = _require_request_endpoint(api_url, api_key, "DeepSeek")
    except ValueError as e:
        return _json_reply_message(str(e), str(e))

    client = OpenAI(
        api_key=request_api_key,
        base_url=request_api_url)
    for step in range(_agent_step_limit()):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=conversation_input,
                tools=_active_deepseek_tool_definitions(),
                tool_choice="auto",
                response_format={"type": "json_object"},
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )
        except Exception as e:
            return _json_reply_message(
                f"模型请求失败，请稍后重试。错误：{type(e).__name__}: {e}",
                "モデルのリクエストに失敗しました。少し待ってからもう一度試してください。",
            )

        if isinstance(response, str):
            content = response.strip()
            print(f"deepseek returned raw string chars={len(content)}")
            final_content = _finalize_model_content(content, summary_mode)
            _log_model_reply("deepseek", final_content)
            return {"role": "assistant", "content": final_content}

        if isinstance(response, dict):
            content = _extract_response_text(response).strip()
            if content:
                print("deepseek returned raw dict")
                final_content = _finalize_model_content(content, summary_mode)
                _log_model_reply("deepseek", final_content)
                return {"role": "assistant", "content": final_content}
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")

        if not tool_calls:
            content = _content_to_text(content).strip()
            if not content:
                return {
                    "role": "assistant",
                    "content": _finalize_model_content("", summary_mode),
                }
            final_content = _finalize_model_content(content, summary_mode)
            _log_model_reply("deepseek", final_content)
            return {"role": "assistant", "content": final_content}

        assistant_message = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                _chat_tool_call_to_message(tool_call)
                for tool_call in tool_calls
            ],
        }
        conversation_input.append(assistant_message)

        for tool_call in tool_calls:
            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id is None and isinstance(tool_call, dict):
                tool_call_id = tool_call.get("id")
            conversation_input.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": _run_chat_completion_tool_call(tool_call),
            })

    return _json_reply_message(
        "搜索步骤过多，我先停一下；你可以换个更具体的问题再试。",
        "検索の手順が多すぎたので、いったん止めます。もう少し具体的に聞いてください。",
    )


# 新版 get_answer 使用 Gemini，并保留上面的旧实现便于回退。
def _get_current_gemini_api_key():
    with key_lock:
        if not key_lt:
            return None

        return key_lt[key_index % len(key_lt)]


def _get_gemini_api_key_candidates():
    with key_lock:
        if not key_lt:
            return []

        start_index = key_index % len(key_lt)
        candidates = []
        seen = set()
        for offset in range(len(key_lt)):
            original_index = (start_index + offset) % len(key_lt)
            api_key = key_lt[original_index]
            if not api_key or api_key in seen:
                continue
            seen.add(api_key)
            candidates.append({"index": original_index, "key": api_key})
        return candidates


def _set_gemini_api_key_index(new_index):
    global key_index

    try:
        normalized_index = int(new_index)
    except (TypeError, ValueError):
        return False

    with key_lock:
        if not key_lt or normalized_index < 0 or normalized_index >= len(key_lt):
            return False

        key_index = normalized_index
        return True


def _advance_gemini_api_key():
    global key_index

    with key_lock:
        if not key_lt:
            return None

        key_index = (key_index + 1) % len(key_lt)
        return key_lt[key_index % len(key_lt)]


def _is_resource_exhausted(response):
    if response is None:
        return False

    if response.status_code == 429:
        return True

    try:
        data = response.json()
    except ValueError:
        data = {}

    error = data.get("error") if isinstance(data, dict) else {}
    if not isinstance(error, dict):
        error = {}
    status = str(error.get("status") or "").upper()
    message = str(error.get("message") or "").lower()
    response_text = getattr(response, "text", "") or ""
    response_text = response_text.lower()
    return (
            status == "RESOURCE_EXHAUSTED"
            or "resource exhausted" in message
            or "quota" in message
            or "resource_exhausted" in response_text
            or "resource exhausted" in response_text
            or "quota" in response_text
    )


def _should_rotate_gemini_key(response):
    if response is None:
        return False
    if response.status_code in RETRYABLE_GEMINI_KEY_STATUSES:
        return True
    return _is_resource_exhausted(response)


def _messages_to_gemini_contents(messages):
    contents = []
    system_parts = []

    vision_enabled = _model_supports_vision("gemini", GEMINI_MODEL)

    for message in _iter_messages(messages):
        role = (message.get("role") or "user").lower()
        content = message.get("content")
        if not content:
            continue

        if role in ("system", "developer"):
            system_text = _content_to_ocr_text(content).strip()
            if system_text:
                system_parts.append({"text": system_text})
        else:
            gemini_role = "model" if role == "assistant" else "user"
            if role in ("assistant", "model"):
                serialized = _content_to_ocr_text(content).strip()
                if not serialized:
                    continue
                parts = [{"text": serialized}]
            else:
                serialized = _serialize_content_for_model(content, "gemini", vision_enabled)
                if not serialized:
                    continue
                if isinstance(serialized, list):
                    parts = serialized
                else:
                    parts = [{"text": serialized}]
            contents.append({"role": gemini_role, "parts": parts})

    return contents, system_parts


def _extract_gemini_text(data):
    texts = []

    for candidate in data.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if text:
                texts.append(text)

    return "\n".join(texts).strip()


def _extract_gemini_function_calls(data):
    calls = []
    for candidate in data.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            function_call = part.get("functionCall")
            if function_call:
                calls.append(function_call)
    return calls


def _extract_gemini_model_content(data):
    """Return the original model content for a tool-call turn.

    Gemini 3 attaches a thoughtSignature to function-call parts. The complete
    content must be replayed unchanged in the next request or the API rejects
    the function response.
    """
    for candidate in data.get("candidates") or []:
        content = candidate.get("content") or {}
        if any(part.get("functionCall") for part in content.get("parts") or []):
            return content
    return None


def _gemini_tool_result_to_response(name, output):
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        parsed = {"result": output}
    return {
        "functionResponse": {
            "name": name,
            "response": parsed if isinstance(parsed, dict) else {"result": parsed},
        }
    }


def _run_gemini_function_call(function_call):
    name = function_call.get("name")
    arguments = function_call.get("args") or {}
    if not isinstance(arguments, dict):
        return json.dumps({"error": "Invalid Gemini tool arguments."}, ensure_ascii=False)
    return _run_local_tool_by_name(name, arguments)


def _gemini_error_message(error):
    return _json_reply_message(
        f"模型请求超时或连接中断，请稍后重试。错误：{type(error).__name__}: {error}",
        "モデルの接続が中断されました。少し待ってからもう一度試してください。",
    )


def _gemini_connection_interrupted(error=None, context="gemini request failed"):
    return _json_reply_message(
        "连接中断，请稍后重试。",
        "接続が中断されました。少し待ってからもう一度試してください。",
    )


def get_answer_gemini(api_url, api_key, messages):
    contents, system_parts = _messages_to_gemini_contents(messages)
    summary_mode = _is_summary_turn(
        contents + [{"role": "developer", "content": part.get("text", "")} for part in system_parts])
    if not contents:
        return _json_reply_message(
            "没有收到有效的用户消息。",
            "有効なユーザーのメッセージを受け取れませんでした。",
        )

    conversation_contents = list(contents)
    agent_step = 0

    try:
        request_api_url, request_api_key = _require_request_endpoint(api_url, api_key, "Gemini")
    except ValueError as e:
        return _json_reply_message(str(e), str(e))

    gemini_api_url = _gemini_api_url(request_api_url, GEMINI_MODEL)

    last_error = None
    total_keys = 1
    max_consecutive_errors = total_keys + 1
    consecutive_errors = 0

    while consecutive_errors < max_consecutive_errors:
        try:
            payload = {
                "contents": conversation_contents,
                "tools": _active_gemini_tool_declarations(),
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "AUTO",
                        "allowedFunctionNames": _active_gemini_function_names(),
                    },
                },
                "generationConfig": _gemini_generation_config(GEMINI_MODEL),
            }
            if system_parts:
                payload["systemInstruction"] = {"parts": system_parts}

            response = requests.post(
                gemini_api_url,
                params={"key": request_api_key},
                json=payload,
                timeout=MODEL_TIMEOUT_SECONDS,
            )
            if response.status_code == 503:
                last_error = RuntimeError(f"Gemini service unavailable: {response.text[:300]}")
                break

            response.raise_for_status()
            data = response.json()
            function_calls = _extract_gemini_function_calls(data)

            if function_calls:
                if agent_step >= _agent_step_limit():
                    return _json_reply_message(
                        "工具调用步骤过多，我先停一下；你可以换个更具体的问题再试。",
                        "ツール呼び出しが多すぎたので、いったん止めます。もう少し具体的に聞いてください。",
                    )

                agent_step += 1
                model_content = _extract_gemini_model_content(data)
                if not model_content:
                    raise ValueError("Gemini function call is missing its model content.")
                conversation_contents.append(model_content)
                conversation_contents.append({
                    "role": "user",
                    "parts": [
                        _gemini_tool_result_to_response(
                            function_call.get("name"),
                            _run_gemini_function_call(function_call),
                        )
                        for function_call in function_calls
                    ],
                })
                continue

            content = _extract_gemini_text(data)

            if content:
                final_content = _finalize_model_content(content, summary_mode)
                _log_model_reply("gemini", final_content)
                return {"role": "assistant", "content": final_content}

            finish_reason = ""
            candidates = data.get("candidates") or []
            if candidates:
                finish_reason = candidates[0].get("finishReason") or ""

            return _json_reply_message(
                f"模型返回为空。finishReason: {finish_reason or 'UNKNOWN'}",
                "モデルの返答が空でした。少し待ってからもう一度試してください。",
            )
        except requests.HTTPError as e:
            last_error = e
            if _should_rotate_gemini_key(e.response):
                consecutive_errors += 1
                continue
            break
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            break
        except ValueError as e:
            last_error = e
            break
        except Exception as e:
            last_error = e
            break

    return _gemini_connection_interrupted(last_error or RuntimeError("Gemini request failed"))


def get_answer(model, api_url, api_key, messages):
    if model in keywords_to_model.keys():
        endpoint_token = _set_request_endpoint(api_url, api_key)
        model_token = _set_request_model_name(model)
        try:
            return keywords_to_model[model](api_url, api_key, messages)
        finally:
            _reset_request_model_name(model_token)
            _reset_request_endpoint(endpoint_token)
    else:
        return False


keywords_to_model = {
    "gpt": get_answer_gpt,
    "gemini": get_answer_gemini,
    "deepseek": get_ans_deepseek,

}
