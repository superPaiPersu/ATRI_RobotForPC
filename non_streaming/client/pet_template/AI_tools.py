from __future__ import annotations

import base64
import json
import mimetypes
import re
import socketio
from pathlib import Path


AI_FILE_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".html",
    ".css",
    ".scss",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".bat",
    ".ps1",
    ".sh",
    ".sql",
    ".log",
    ".rpy"
}

AI_FILE_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
}

AI_FILE_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[\"']?[^\"'\s]{8,}"),
]


def _file_kind(extension: str) -> str:
    if extension in AI_FILE_IMAGE_EXTENSIONS:
        return "image"
    if extension in AI_FILE_TEXT_EXTENSIONS:
        return "text"
    return "unsupported"


def _decode_file_text(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "cp936"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _redact_file_secrets(text: str) -> str:
    for pattern in AI_FILE_SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _prepare_file_for_ai(
    file_path: str,
    max_chars: int = 5000000,
    max_bytes: int = 200 * 1024 * 1024,
    allow_unsupported: bool = False,
    redact_secrets: bool = True,
) -> dict:
    raw_path = str(file_path or "").strip()
    if not raw_path:
        return {"ok": False, "error": "file_path is required"}

    try:
        path = Path(raw_path).expanduser().resolve()
    except OSError as exc:
        return {"ok": False, "error": f"Invalid file path: {exc}"}

    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    extension = path.suffix.lower()
    kind = _file_kind(extension)
    if not allow_unsupported and kind == "unsupported":
        return {"ok": False, "error": f"Unsupported file extension: {extension}"}

    try:
        size = path.stat().st_size
    except OSError as exc:
        return {"ok": False, "error": f"Cannot stat file: {exc}"}


    if size > max_bytes:
        return {
            "ok": False,
            "error": "File is too large to read directly.",
            "path": str(path),
            "size_bytes": size,
            "max_bytes": max_bytes,
        }

    if kind == "image":
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return {"ok": False, "error": f"Cannot read file: {exc}"}

        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
        text = (
            "The user uploaded a local image file.\n"
            f"File name: {path.name}\n"
            f"Original path: {path}\n"
            f"MIME type: {mime_type}\n"
            f"Size bytes: {size}\n"
            "Please inspect the image directly and answer the user."
        )

        return {
            "ok": True,
            "kind": "image",
            "path": str(path),
            "name": path.name,
            "extension": extension,
            "size_bytes": size,
            "mime_type": mime_type,
            "prompt_message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
                "file_upload": {
                    "kind": "image",
                    "name": path.name,
                    "extension": extension,
                    "mime_type": mime_type,
                    "size_bytes": size,
                },
            },
        }

    try:
        text, encoding = _decode_file_text(path.read_bytes())
    except OSError as exc:
        return {"ok": False, "error": f"Cannot read file: {exc}"}

    if redact_secrets:
        text = _redact_file_secrets(text)


    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    note = "以下是用户主动选择的本地文件内容。"
    if truncated:
        note += "内容已按客户端限制截断。"
    prompt = (
        f"{note}\n"
        f"文件名: {path.name}\n"
        f"原始路径: {path}\n"
        f"编码: {encoding}\n"
        f"字符数: {len(text)}\n\n"
        f"```{extension.lstrip('.')}\n{text}\n```"
    )

    return {
        "ok": True,
        "kind": "text",
        "path": str(path),
        "name": path.name,
        "extension": extension,
        "size_bytes": size,
        "encoding": encoding,
        "content": text,
        "content_chars": len(text),
        "truncated": truncated,
        "redacted": bool(redact_secrets),
        "prompt_message": {"role": "user", "content": prompt},
    }


class SocketAI:
    """WebSocket client wrapper for blocking request/response calls."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000", username: str = "", pet_id: str = ""):
        self.base_url = base_url
        self.username = username
        self.model_name=""
        self.api_url=""
        self.api_key=""
        self.pet_id = pet_id
        print(self.pet_id)
        self.sio = socketio.Client()
        self._connect()

    def _connect(self):
        if self.sio.connected:
            return
        self.sio.connect(self.base_url, wait_timeout=10)

    def _ensure_connected(self):
        if not self.sio.connected or "/" not in self.sio.namespaces:
            try:
                self.sio.disconnect()
            except Exception:
                pass
            self._connect()

    def emit(self, event: str, data: dict, callback=None):
        """Send a fire-and-forget event with the username injected."""
        if self.sio.connected:
            data["username"] = self.username
            self.sio.emit(event, data, callback=callback)

    def wait_for_response(self, target: str, data, timeout: int = 100,special=None):
        payload = {
            "data": data,
            "username": self.username,
            "model_name": self.model_name,
            "pet_id": self.pet_id,
            "api_url":self.api_url,
            "api_key":self.api_key,
            # "api_url":"https://api.yomiapi.com/v1",
            "special":special,
        }
        self._ensure_connected()
        try:
            return self.sio.call(target, payload, timeout=timeout)
        except (socketio.exceptions.BadNamespaceError, socketio.exceptions.DisconnectedError):
            self._ensure_connected()
            return self.sio.call(target, payload, timeout=timeout)

    def save_dialogue_history(self,history):
        return self.wait_for_response("save_dialogue_history",history)
    def update_summary_history(self,history,chat_begin_time):
        print("sendoi")
        return self.wait_for_response("update_summary_history",history,special=chat_begin_time)

    def get_dialogue_history(self):
        return self.wait_for_response("get_dialogue_history","")
    def get_summary_history(self):
        return self.wait_for_response("get_summary_history","")
    def translate(self, audio_data):
        return self.wait_for_response("audio_translate", audio_data)

    def get_answer(self, chat_history):

        return self.wait_for_response("get_answer", chat_history)

    def upload_file(
        self,prompt,
        file_path: str,
        max_chars: int = 5000000,
        max_bytes: int = 200 * 1024 * 1024,
        allow_unsupported: bool = False,
        redact_secrets: bool = True,
    ):
        ans=_prepare_file_for_ai(
            file_path,
            max_chars=max_chars,
            max_bytes=max_bytes,
            allow_unsupported=allow_unsupported,
            redact_secrets=redact_secrets,
        )
        if not ans.get("ok"):
            return {
                "content": json.dumps({
                    "action": "reply",
                    "cn": ans.get("error", "读取文件失败"),
                    "jp": ans.get("error", "读取文件失败"),
                    "mood": "normal",
                    "expression": "normal"
                }, ensure_ascii=False)
            }


        content= [ans["prompt_message"]]
        return self.get_answer(prompt+content)

    def change_model(self,is_serious):
        return self.wait_for_response("change_model", is_serious)

    def voice_produce(self, dt):
        payload = dict(dt)
        if self.pet_id:
            payload["pet_id"] = self.pet_id
        return self.wait_for_response("voice_produce", payload)
