from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import socketio
import sys
import threading
import time
import uuid
from pathlib import Path
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

try:
    from .harness_bridge import run_history_search, _concept_tags, merge_saved_history
except ImportError:
    from harness_bridge import run_history_search, _concept_tags, merge_saved_history


def _configure_network_logger():
    logger = logging.getLogger("ai_pet.client.network")
    if getattr(logger, "_configured", False):
        return logger

    level_name = os.getenv("AI_PET_CLIENT_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    default_log_base = Path(
        sys.executable if getattr(sys, "frozen", False) else __file__
    ).resolve().parent
    log_path = Path(
        os.getenv(
            "AI_PET_CLIENT_LOG_FILE",
            str(default_log_base / "client_network.log"),
        )
    ).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except OSError:
        # The desktop client should still run if its directory is read-only.
        pass

    logger._configured = True
    return logger


NETWORK_LOGGER = _configure_network_logger()


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
        note += " 内容已按客户端限制截断。"
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
    """WebSocket client wrapper for streaming chat and blocking utility calls."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5000",
        username: str = "",
        pet_id: str = "",
        client_type: str = "mobile",
        chunk_sink=None,
        realtime_chunk_sink=None,
        realtime_event_sink=None,
        history_search_callback=None,
        conversation_action_callback=None,
    ):
        self.base_url = base_url
        self.username = username
        self.model_name=""
        self.api_url=""
        self.api_key=""
        self.realtime_url = os.getenv(
            "QWEN_REALTIME_URL",
            "wss://llm-umoeb8yx4wcyqwo0.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        ).strip()
        self.realtime_api_key = os.getenv("QWEN_REALTIME_API_KEY", "").strip()
        self.client_type = (
            os.getenv("AI_PET_CLIENT_TYPE", client_type).strip().lower()
            or client_type
        )
        self.conversation_id = uuid.uuid4().hex
        self.pet_id = pet_id
        self._chunk_sink = chunk_sink
        self._realtime_chunk_sink = realtime_chunk_sink
        self._realtime_event_sink = realtime_event_sink
        if history_search_callback is not None and not callable(history_search_callback):
            raise TypeError("history_search_callback must be callable or None")
        self._history_search_callback = history_search_callback
        self._history_search_lock = threading.RLock()
        if conversation_action_callback is not None and not callable(conversation_action_callback):
            raise TypeError("conversation_action_callback must be callable or None")
        self._conversation_action_callback = conversation_action_callback
        self._conversation_action_lock = threading.RLock()
        self._conversation_action_results = {}
        self._realtime_active = False
        self._realtime_response_active = False
        self._realtime_trigger_pending = False
        self._realtime_heartbeat_seconds = None
        self._realtime_state_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._stream_active = False
        self._stream_started_at = None
        self._stream_chunk_count = 0
        self._realtime_audio_count = 0
        self._realtime_audio_bytes = 0
        self._realtime_audio_log_count = 0
        self._realtime_audio_started_at = None
        self._realtime_chunk_count = 0
        self._realtime_event_count = 0
        print(self.pet_id)
        self.sio = socketio.Client()
        self.sio.on("connect", self._on_socket_connect)
        self.sio.on("disconnect", self._on_socket_disconnect)
        self.sio.on("chat_chunk", self._on_chat_chunk)
        self.sio.on("realtime_chat_chunk", self._on_realtime_chat_chunk)
        self.sio.on("realtime_chat_event", self._on_realtime_chat_event)
        self.sio.on("dialogue_history_request", self._on_dialogue_history_request)
        self.sio.on("conversation_action_request", self._on_conversation_action_request)
        self._connect()

    def set_history_search_callback(self, callback):
        """Register a live message or session-index callback on a Socket.IO worker."""
        if callback is not None and not callable(callback):
            raise TypeError("History search callback must be callable or None")
        with self._history_search_lock:
            self._history_search_callback = callback

    def _on_dialogue_history_request(self, data):
        """Return a Socket.IO acknowledgement, separate from chat output."""
        request_id = data.get("request_id") if isinstance(data, dict) else None
        started = time.monotonic()
        try:
            if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
                raise ValueError("Invalid request_id")
            if data.get("username") != self.username or data.get("pet_id") != self.pet_id:
                raise ValueError("History request identity mismatch")
            tags = _concept_tags(data.get("tags"))
            with self._history_search_lock:
                if self._history_search_callback is None:
                    raise RuntimeError("History search callback is not registered")
                result = run_history_search(self._history_search_callback, tags)
            messages = result["messages"]
            response = {"request_id": request_id, "messages": messages, "matched": bool(messages)}
            if len(json.dumps(response, ensure_ascii=False).encode("utf-8")) > 2 * 1024 * 1024:
                raise ValueError("History result exceeds 2 MiB")
            NETWORK_LOGGER.info("history_search_done request=%s messages=%d elapsed_ms=%.1f", request_id[:12], len(messages), (time.monotonic() - started) * 1000)
            return response
        except Exception as exc:
            NETWORK_LOGGER.warning("history_search_failed request=%s error_type=%s elapsed_ms=%.1f", str(request_id)[:12], type(exc).__name__, (time.monotonic() - started) * 1000)
            return {"request_id": request_id, "error": "Local history search failed: " + type(exc).__name__}

    def set_conversation_action_callback(self, callback):
        """Register main.py's save_log_async(is_exit)."""
        if callback is not None and not callable(callback):
            raise TypeError("Conversation action callback must be callable or None")
        with self._conversation_action_lock:
            self._conversation_action_callback = callback

    def _on_conversation_action_request(self, data):
        request_id = data.get("request_id") if isinstance(data, dict) else None
        try:
            if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
                raise ValueError("Invalid request_id")
            if data.get("username") != self.username or data.get("pet_id") != self.pet_id:
                raise ValueError("Conversation action identity mismatch")
            action = data.get("action")
            if action not in ("save", "conversation-over"):
                raise ValueError("Unknown conversation action")
            with self._conversation_action_lock:
                previous = self._conversation_action_results.get(request_id)
                if previous is not None:
                    if previous["action"] != action:
                        raise ValueError("request_id was already used for another action")
                    return dict(previous)
                if self._conversation_action_callback is None:
                    raise RuntimeError("Conversation action callback is not registered")
                accepted = self._conversation_action_callback(action == "conversation-over")
                if accepted is False:
                    raise RuntimeError("Client rejected the conversation action")
                result = {"request_id": request_id, "ok": True, "status": "scheduled", "action": action}
                self._conversation_action_results[request_id] = result
                if len(self._conversation_action_results) > 128:
                    self._conversation_action_results.pop(next(iter(self._conversation_action_results)))
                NETWORK_LOGGER.info("conversation_action_scheduled request=%s action=%s", request_id[:12], action)
                return dict(result)
        except Exception as exc:
            NETWORK_LOGGER.warning("conversation_action_failed request=%s error_type=%s", str(request_id)[:12], type(exc).__name__)
            return {"request_id": request_id, "ok": False, "error": "Local conversation action failed: " + type(exc).__name__}

    def _socket_transport(self):
        try:
            return str(self.sio.transport())
        except Exception:
            try:
                return str(self.sio.eio.transport)
            except Exception:
                return "unknown"

    def _on_socket_connect(self):
        NETWORK_LOGGER.info(
            "socket_connected base_url=%s transport=%s",
            urlparse(self.base_url).netloc or self.base_url,
            self._socket_transport(),
        )

    def _on_socket_disconnect(self):
        NETWORK_LOGGER.info("socket_disconnected transport=%s", self._socket_transport())

    def _connect(self):
        if self.sio.connected:
            return
        started_at = time.monotonic()
        NETWORK_LOGGER.info(
            "socket_connect_begin base_url=%s timeout_s=10",
            urlparse(self.base_url).netloc or self.base_url,
        )
        try:
            self.sio.connect(self.base_url, wait_timeout=10)
        except Exception as exc:
            NETWORK_LOGGER.error(
                "socket_connect_failed elapsed_ms=%.1f error=%s: %s",
                (time.monotonic() - started_at) * 1000,
                type(exc).__name__,
                exc,
            )
            raise
        NETWORK_LOGGER.info(
            "socket_connect_done elapsed_ms=%.1f transport=%s connected=%s",
            (time.monotonic() - started_at) * 1000,
            self._socket_transport(),
            self.sio.connected,
        )

    def _ensure_connected(self):
        if not self.sio.connected or "/" not in self.sio.namespaces:
            NETWORK_LOGGER.warning(
                "socket_reconnect_needed connected=%s namespaces=%s",
                self.sio.connected,
                sorted(self.sio.namespaces),
            )
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

    def set_chunk_sink(self, chunk_sink):
        self._chunk_sink = chunk_sink

    def set_realtime_sinks(self, chunk_sink=None, event_sink=None):
        self._realtime_chunk_sink = chunk_sink
        self._realtime_event_sink = event_sink

    def _push_chunk(self, flag, payload):
        sink = self._chunk_sink
        if sink:
            sink([flag, payload])

    def _on_chat_chunk(self, data):
        if not self._stream_active:
            return

        self._stream_chunk_count += 1

        if isinstance(data, (list, tuple)) and len(data) >= 2:
            flag, payload = data[0], data[1]
        elif isinstance(data, dict):
            flag = data.get("flag")
            payload = data.get("payload")
        else:
            return

        self._push_chunk(flag, payload)
        if self._stream_chunk_count == 1:
            NETWORK_LOGGER.info(
                "chat_first_chunk elapsed_ms=%.1f flag=%s",
                (time.monotonic() - self._stream_started_at) * 1000
                if self._stream_started_at is not None
                else -1,
                flag,
            )
        if flag == "end":
            NETWORK_LOGGER.info(
                "chat_stream_done elapsed_ms=%.1f chunks=%d",
                (time.monotonic() - self._stream_started_at) * 1000
                if self._stream_started_at is not None
                else -1,
                self._stream_chunk_count,
            )
            self._stream_active = False

    @staticmethod
    def _unpack_chunk(data):
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            return data[0], data[1]
        if isinstance(data, dict):
            return data.get("flag"), data.get("payload")
        return None, None

    def _on_realtime_chat_chunk(self, data):
        flag, payload = self._unpack_chunk(data)
        if flag is None:
            return
        self._realtime_chunk_count += 1
        if (
            self._realtime_chunk_count == 1
            or flag in ("params", "end", "cancel")
            or self._realtime_chunk_count % 100 == 0
        ):
            NETWORK_LOGGER.info(
                "realtime_chunk_received count=%d flag=%s payload_type=%s",
                self._realtime_chunk_count,
                flag,
                type(payload).__name__,
            )
        if flag == "params" and isinstance(payload, dict):
            action = str(payload.get("action") or "reply").strip()
            if action == "reply":
                raw_seconds = payload.get("heart_beat_seconds")
                try:
                    seconds = float(raw_seconds) if raw_seconds not in (None, "") else 30.0
                    with self._realtime_state_lock:
                        self._realtime_heartbeat_seconds = max(5.0, seconds)
                except (TypeError, ValueError):
                    pass
        sink = self._realtime_chunk_sink
        if sink:
            sink([flag, payload])

    def _on_realtime_chat_event(self, data):
        if isinstance(data, dict):
            event_type = str(data.get("type") or "").strip()
            self._realtime_event_count += 1
            server_sent_at = data.get("_server_sent_at")
            server_to_client_ms = None
            try:
                server_to_client_ms = (time.time() - float(server_sent_at)) * 1000
            except (TypeError, ValueError):
                pass
            NETWORK_LOGGER.info(
                "realtime_event_received count=%d type=%s turn_id=%s server_to_client_ms=%s",
                self._realtime_event_count,
                event_type,
                data.get("turn_id"),
                f"{server_to_client_ms:.1f}" if server_to_client_ms is not None else "unknown",
            )
            with self._realtime_state_lock:
                if event_type == "assistant_started":
                    self._realtime_response_active = True
                    self._realtime_trigger_pending = False
                elif event_type in (
                    "turn_completed",
                    "turn_cancelled",
                    "error",
                    "closed",
                ):
                    self._realtime_response_active = False
                    self._realtime_trigger_pending = False
            if event_type in ("error", "closed"):
                self._realtime_active = False
        sink = self._realtime_event_sink
        if sink and isinstance(data, dict):
            sink(data)

    def wait_for_response(self, target: str, data, timeout: int = 100,special=None):
        payload = {
            "data": data,
            "username": self.username,
            "model_name": self.model_name,
            "pet_id": self.pet_id,
            "api_url":self.api_url,
            "api_key":self.api_key,
            "special":special,
        }
        self._ensure_connected()
        started_at = time.monotonic()
        NETWORK_LOGGER.info("socket_call_begin event=%s timeout_s=%s", target, timeout)
        try:
            response = self.sio.call(target, payload, timeout=timeout)
            NETWORK_LOGGER.info(
                "socket_call_done event=%s elapsed_ms=%.1f response_type=%s",
                target,
                (time.monotonic() - started_at) * 1000,
                type(response).__name__,
            )
            return response
        except (socketio.exceptions.BadNamespaceError, socketio.exceptions.DisconnectedError):
            NETWORK_LOGGER.warning(
                "socket_call_retry event=%s elapsed_ms=%.1f",
                target,
                (time.monotonic() - started_at) * 1000,
            )
            self._ensure_connected()
            retry_started_at = time.monotonic()
            response = self.sio.call(target, payload, timeout=timeout)
            NETWORK_LOGGER.info(
                "socket_call_retry_done event=%s elapsed_ms=%.1f response_type=%s",
                target,
                (time.monotonic() - retry_started_at) * 1000,
                type(response).__name__,
            )
            return response
        except Exception as exc:
            NETWORK_LOGGER.error(
                "socket_call_failed event=%s elapsed_ms=%.1f error=%s: %s",
                target,
                (time.monotonic() - started_at) * 1000,
                type(exc).__name__,
                exc,
            )
            raise

    def save_dialogue_history(self,history):
        response = self.wait_for_response("save_dialogue_history", history, special={"return_saved_history": True})
        if isinstance(response, dict) and response.get("status") == "success":
            try:
                with self._history_search_lock:
                    if self._history_search_callback is not None:
                        merge_saved_history(self._history_search_callback, response["saved_history"], response["tags"])
            except Exception as exc:
                # The server has already persisted this batch. An index error
                # must not turn it into a failed save and trigger duplicate writes.
                NETWORK_LOGGER.warning("history_index_refresh_failed error_type=%s", type(exc).__name__)
            return "success"
        return response
    def update_summary_history(self,history,chat_begin_time):
        print("sendoi")
        return self.wait_for_response("update_summary_history",history,special=chat_begin_time)

    def get_dialogue_history(self):
        return self.wait_for_response("get_dialogue_history","")

    def get_dialogue_tags(self):
        return self.wait_for_response("get_dialogue_tags","")
    def get_summary_history(self):
        return self.wait_for_response("get_summary_history","")
    def translate(self, audio_data):
        return self.wait_for_response("audio_translate", audio_data)

    def ja2cn(self, text):
        response = self.wait_for_response("ja2cn", {"text": text})
        if isinstance(response, dict):
            return str(response.get("translated_text") or "").strip()
        return ""

    def get_answer(self, chat_history):
        payload = {
            "dialogue_history": self._history_search_callback is not None,
            "conversation_actions": self._conversation_action_callback is not None,
            "data": chat_history,
            "username": self.username,
            "client_type": self.client_type,
            "conversation_id": self.conversation_id,
            "model_name": self.model_name,
            "pet_id": self.pet_id,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "special": None,
        }
        self._ensure_connected()
        with self._stream_lock:
            if self._stream_active:
                return False

            self._stream_active = True
            self._stream_started_at = time.monotonic()
            self._stream_chunk_count = 0
            NETWORK_LOGGER.info(
                "chat_stream_begin history_items=%d transport=%s",
                len(chat_history or []),
                self._socket_transport(),
            )
            try:
                # Chat is event-driven: the server sends params/text/end as
                # chat_chunk events. Do not wait for the handler's final ack.
                self.sio.emit("get_answer", payload)
            except Exception:
                self._stream_active = False
                self._stream_started_at = None
                raise

        return None
    def start_realtime(
        self,
        history,
        model=None,
        transcription_model=None,
        provider="qwen",
        realtime_url=None,
        realtime_api_key=None,
    ):
        provider = str(provider or "minicpm").strip().lower()
        default_model = (
            "qwen-audio-3.0-realtime-plus"
            if provider == "qwen"
            else "MiniCPM-o-4_5-int4" if provider in ("minicpm", "minicpm-o", "minicpm_local", "local")
            else "gpt-realtime"
        )
        if realtime_url is None:
            realtime_url = (
                self.realtime_url
                if provider == "qwen"
                else self.api_url
            )
        if realtime_api_key is None:
            realtime_api_key = (
                self.realtime_api_key
                if provider == "qwen"
                else self.api_key
            )

        payload = {
            "data": history,
            "dialogue_history": self._history_search_callback is not None,
            "conversation_actions": self._conversation_action_callback is not None,
            "username": self.username,
            "client_type": self.client_type,
            "conversation_id": self.conversation_id,
            "pet_id": self.pet_id,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "realtime_provider": provider,
            "realtime_url": realtime_url,
            "realtime_api_key": realtime_api_key,
            "realtime_model": model or default_model,
            "model_name": model or default_model,
            "transcription_model": transcription_model,
        }
        self._ensure_connected()
        with self._realtime_state_lock:
            self._realtime_response_active = False
            self._realtime_trigger_pending = False
            self._realtime_heartbeat_seconds = None
        self._realtime_audio_count = 0
        self._realtime_audio_bytes = 0
        self._realtime_audio_log_count = 0
        self._realtime_audio_started_at = time.monotonic()
        self._realtime_chunk_count = 0
        self._realtime_event_count = 0
        started_at = time.monotonic()
        NETWORK_LOGGER.info(
            "realtime_start_begin provider=%s model=%s history_items=%d transport=%s",
            provider,
            payload["realtime_model"],
            len(payload["data"] or []),
            self._socket_transport(),
        )
        # MiniCPM-o 4.5 may need roughly 45 seconds for its first GPU load.
        try:
            response = self.sio.call("start_realtime", payload, timeout=120)
        except Exception as exc:
            NETWORK_LOGGER.error(
                "realtime_start_failed elapsed_ms=%.1f error=%s: %s",
                (time.monotonic() - started_at) * 1000,
                type(exc).__name__,
                exc,
            )
            raise
        NETWORK_LOGGER.info(
            "realtime_start_done elapsed_ms=%.1f ok=%s server_elapsed_ms=%s server_received_at=%s server_ready_at=%s",
            (time.monotonic() - started_at) * 1000,
            bool(isinstance(response, dict) and response.get("ok")),
            response.get("server_elapsed_ms") if isinstance(response, dict) else None,
            response.get("server_received_at") if isinstance(response, dict) else None,
            response.get("server_ready_at") if isinstance(response, dict) else None,
        )
        if isinstance(response, dict) and response.get("ok"):
            self._realtime_active = True
        else:
            with self._realtime_state_lock:
                self._realtime_response_active = False
                self._realtime_trigger_pending = False
                self._realtime_heartbeat_seconds = None
        return response

    def send_realtime_audio(self, pcm16_bytes, sample_rate=16000):
        if not self._realtime_active:
            return False
        if not isinstance(pcm16_bytes, (bytes, bytearray)):
            raise TypeError("Realtime audio must be PCM16 bytes")
        self._realtime_audio_count += 1
        self._realtime_audio_bytes += len(pcm16_bytes)
        if (
            self._realtime_audio_count == 1
            or self._realtime_audio_count - self._realtime_audio_log_count >= 100
        ):
            now = time.monotonic()
            NETWORK_LOGGER.info(
                "realtime_audio_emit count=%d bytes_total=%d interval_ms=%.1f transport=%s",
                self._realtime_audio_count,
                self._realtime_audio_bytes,
                (now - self._realtime_audio_started_at) * 1000
                if self._realtime_audio_started_at is not None
                else -1,
                self._socket_transport(),
            )
            self._realtime_audio_log_count = self._realtime_audio_count
        self.emit("realtime_audio", {
            "audio": base64.b64encode(bytes(pcm16_bytes)).decode("ascii"),
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        })
        return True

    def stop_realtime(self):
        if not self._realtime_active:
            return {"ok": True, "active": False}
        started_at = time.monotonic()
        NETWORK_LOGGER.info("realtime_stop_begin audio_count=%d", self._realtime_audio_count)
        try:
            response = self.sio.call(
                "stop_realtime",
                {"username": self.username, "pet_id": self.pet_id},
                timeout=10,
            )
            NETWORK_LOGGER.info(
                "realtime_stop_done elapsed_ms=%.1f ok=%s server_elapsed_ms=%s",
                (time.monotonic() - started_at) * 1000,
                bool(isinstance(response, dict) and response.get("ok")),
                response.get("server_elapsed_ms") if isinstance(response, dict) else None,
            )
        except Exception as exc:
            NETWORK_LOGGER.error(
                "realtime_stop_failed elapsed_ms=%.1f error=%s: %s",
                (time.monotonic() - started_at) * 1000,
                type(exc).__name__,
                exc,
            )
            raise
        finally:
            self._realtime_active = False
            with self._realtime_state_lock:
                self._realtime_response_active = False
                self._realtime_trigger_pending = False
                self._realtime_heartbeat_seconds = None
        return response

    def get_realtime_heartbeat_seconds(self):
        with self._realtime_state_lock:
            return self._realtime_heartbeat_seconds

    def realtime_response_busy(self):
        with self._realtime_state_lock:
            return (
                self._realtime_response_active
                or self._realtime_trigger_pending
            )

    def request_realtime_response(self):
        """Ask the current realtime session to generate a proactive reply."""
        if not self._realtime_active:
            return {"ok": False, "error": "Realtime session is not active"}

        with self._realtime_state_lock:
            if self._realtime_response_active or self._realtime_trigger_pending:
                return {"ok": False, "busy": True, "error": "Realtime response is busy"}
            self._realtime_trigger_pending = True

        payload = {
            "username": self.username,
            "client_type": self.client_type,
            "pet_id": self.pet_id,
        }
        try:
            self._ensure_connected()
            started_at = time.monotonic()
            NETWORK_LOGGER.info("realtime_response_begin")
            response = self.sio.call(
                "realtime_response",
                payload,
                timeout=10,
            )
            NETWORK_LOGGER.info(
                "realtime_response_done elapsed_ms=%.1f ok=%s server_elapsed_ms=%s",
                (time.monotonic() - started_at) * 1000,
                bool(isinstance(response, dict) and response.get("ok")),
                response.get("server_elapsed_ms") if isinstance(response, dict) else None,
            )
        except Exception:
            with self._realtime_state_lock:
                self._realtime_trigger_pending = False
            raise

        if not isinstance(response, dict) or not response.get("ok"):
            with self._realtime_state_lock:
                self._realtime_trigger_pending = False
        return response

    def toggle_realtime(self, flag, rule=None, history=None):
        if flag:
            return self.start_realtime(history or [])
        return self.stop_realtime()
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
                    "cn": ans.get("error", "璇诲彇鏂囦欢澶辫触"),
                    "jp": ans.get("error", "璇诲彇鏂囦欢澶辫触"),
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
