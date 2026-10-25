"""Local MiniCPM-o audio session.

The model is loaded lazily because it is large and should not block normal
server startup.  The session keeps the existing Socket.IO protocol: PCM16
frames go in and text protocol chunks come out.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from server.logging_utils import get_logger


LOGGER = get_logger("minicpm")
DEFAULT_MODEL_PATH = (
    Path("/mnt/storage/ai-pet-model/MiniCPM-o-4_5-int4")
)
LOCAL_MODEL_PATH = Path(
    os.getenv("MINICPM_MODEL_PATH", str(DEFAULT_MODEL_PATH))
).expanduser()
LOCAL_SAMPLE_RATE = 16000
LOCAL_MAX_AUDIO_SECONDS = float(os.getenv("MINICPM_MAX_AUDIO_SECONDS", "30"))
LOCAL_MAX_NEW_TOKENS = int(os.getenv("MINICPM_MAX_NEW_TOKENS", "96"))
LOCAL_SILENCE_SECONDS = float(os.getenv("MINICPM_SILENCE_SECONDS", "0.75"))
LOCAL_VAD_THRESHOLD = float(os.getenv("MINICPM_VAD_THRESHOLD", "0.012"))
LOCAL_GPU_MEMORY = os.getenv("MINICPM_GPU_MEMORY", "22GiB")
LOCAL_CPU_MEMORY = os.getenv("MINICPM_CPU_MEMORY", "24GiB")
LOCAL_MIN_GPU_MEMORY = int(os.getenv("MINICPM_MIN_GPU_MEMORY_MB", "10000"))


def _prepare_transformers_cache():
    """Give Transformers a writable cache and keep all local model code together."""
    cache_root = Path(
        os.getenv(
            "MINICPM_CACHE_DIR",
            str(Path(os.getenv("TEMP", ".")) / "ai_pet_minicpm_cache"),
        )
    ).expanduser()
    modules_cache = cache_root / "modules"
    modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))

    # Transformers copies only directly imported remote-code files. MiniCPM's
    # processor imports another local file one level later, so pre-copy all
    # model Python files to avoid a partial dynamic-module cache on Windows.
    # The sanitization rule changed between Transformers releases, so prepare
    # both cache names used by the supported 4.44 and newer versions.
    safe_names = {
        LOCAL_MODEL_PATH.name,
        LOCAL_MODEL_PATH.name.replace("-", "_hyphen_"),
    }
    for safe_name in safe_names:
        module_dir = modules_cache / "transformers_modules" / safe_name
        module_dir.mkdir(parents=True, exist_ok=True)
        for source in LOCAL_MODEL_PATH.glob("*.py"):
            target = module_dir / source.name
            if not target.exists() or source.stat().st_mtime_ns > target.stat().st_mtime_ns:
                shutil.copy2(source, target)
        (module_dir / "__init__.py").touch(exist_ok=True)


def _repair_offload_hooks(model):
    """Avoid Accelerate CPU-offload hooks on parameterless audio poolers."""
    try:
        from accelerate.hooks import remove_hook_from_module
    except Exception:
        return

    for module in model.modules():
        has_parameters = next(module.parameters(recurse=True), None) is not None
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        if not has_parameters:
            remove_hook_from_module(module, recurse=False)
            continue
        previous = getattr(hook, "prev_module_hook", None)
        previous_model = getattr(previous, "model", None)
        if previous_model is not None and next(previous_model.parameters(recurse=True), None) is None:
            hook.prev_module_hook = None


def _check_local_runtime():
    """Reject configurations known to require unsafe CPU offload."""
    import platform
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MiniCPM local mode requires a CUDA GPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mb = int(free_bytes / (1024 * 1024))
    total_mb = int(total_bytes / (1024 * 1024))
    if platform.system() == "Windows" and free_mb < LOCAL_MIN_GPU_MEMORY:
        raise RuntimeError(
            "MiniCPM local mode was not started: Windows quantized model CPU offload "
            f"is unsafe with only {free_mb} MiB free GPU memory "
            f"({total_mb} MiB total). At least {LOCAL_MIN_GPU_MEMORY} MiB is required; "
            "use the 24 GB Ubuntu server or free GPU memory first."
        )


class MiniCPMLoader:
    """Process-wide lazy model holder."""

    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    _error = None

    @classmethod
    def load(cls):
        if cls._model is not None:
            return cls._model, cls._tokenizer
        with cls._lock:
            if cls._model is not None:
                return cls._model, cls._tokenizer
            if cls._error is not None:
                raise RuntimeError(str(cls._error)) from cls._error
            try:
                if not (LOCAL_MODEL_PATH / "config.json").exists():
                    raise FileNotFoundError(
                        f"MiniCPM-o 4.5 model files not found: {LOCAL_MODEL_PATH}"
                    )
                _check_local_runtime()
                _prepare_transformers_cache()
                import torch
                from transformers import AutoModel

                LOGGER.info("minicpm45_load_begin path=%s", LOCAL_MODEL_PATH)
                model = AutoModel.from_pretrained(
                    str(LOCAL_MODEL_PATH),
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    init_vision=True,
                    init_audio=True,
                    init_tts=False,
                )
                model.eval()
                cls._model = model
                cls._tokenizer = None
                LOGGER.info("minicpm45_load_ready")
                return cls._model, cls._tokenizer
            except Exception as exc:
                cls._error = exc
                LOGGER.exception("minicpm45_load_failed error_type=%s", type(exc).__name__)
                raise


class MiniCPMLocalSession:
    """One long-lived local audio conversation bound to one Socket.IO sid."""

    model = "MiniCPM-o-4_5-int4"
    input_sample_rate = LOCAL_SAMPLE_RATE
    transcription_model = "local-minicpm-audio"

    def __init__(self, messages=None, on_chunk=None, on_event=None, **_kwargs):
        self.on_chunk = on_chunk
        self.on_event = on_event
        self._history = self._normalize_history(messages or [])
        self._queue = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = None
        self._audio = bytearray()
        self._speech_started = False
        self._last_voice_at = None
        self._started_at = time.monotonic()
        self._response_lock = threading.Lock()
        self._response_busy = False

    @staticmethod
    def _normalize_history(messages):
        history = []
        leading_context = []

        def extract_text(value):
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return str(value.get("text") or value.get("content") or "")
            if isinstance(value, list):
                parts = []
                for item in value:
                    text = extract_text(item)
                    if text:
                        parts.append(text)
                return "\n".join(parts)
            return str(value or "")

        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip().lower()
            if role == "model":
                role = "assistant"
            elif role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant"):
                continue
            content = message.get("content")
            if content is None:
                content = message.get("parts", "")
            content = extract_text(content).strip()
            if content:
                if role == "system":
                    leading_context.append(content)
                else:
                    history.append({"role": role, "content": content})

        # Preserve caller-provided context as system data. Do not add a
        # language or response-format instruction here. When the context is
        # too long, keep both ends because the client repeats current rules at
        # the end of the prompt.
        context = "\n\n".join(leading_context).strip()
        context_limit = 24000
        if len(context) > context_limit:
            tail_budget = 12000
            head_budget = context_limit - tail_budget
            context = (
                context[:head_budget]
                + "\n\n[?????????????????]\n\n"
                + context[-tail_budget:]
            )
        # Keep the complete normalized conversation. The old MiniCPM
        # adapter kept only eight messages, which made long voice sessions
        # lose their earlier context.
        recent = list(history)
        while recent and recent[0]["role"] != "user":
            recent.pop(0)
        normalized = []
        if context:
            normalized.append({"role": "system", "content": context})
        return normalized + recent

    @property
    def closed(self):
        return self._closed.is_set()

    def diagnostic_age(self):
        return max(0.0, time.monotonic() - self._started_at)

    def _event(self, payload):
        if self.on_event:
            self.on_event(payload)

    def _chunk(self, flag, payload):
        if self.on_chunk:
            self.on_chunk(flag, payload)

    def start(self, timeout=120):
        self._thread = threading.Thread(
            target=self._run,
            name="minicpm-local-session",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError("MiniCPM local session startup timed out")
        if self._error is not None:
            raise RuntimeError(f"MiniCPM local session failed: {self._error}")
        return self

    def append_audio(self, audio_bytes):
        if self.closed or not audio_bytes:
            return False
        with self._response_lock:
            if self._response_busy:
                # Do not let microphone frames recorded during generation
                # become a stale second turn when the reply finishes.
                return True
        try:
            self._queue.put_nowait(("audio", bytes(audio_bytes)))
            return True
        except queue.Full:
            return False

    def request_response(self, prompt=None):
        if self.closed:
            return {"ok": False, "error": "MiniCPM local session is not active"}
        try:
            self._queue.put_nowait(("response", str(prompt or "")))
            return {"ok": True, "queued": True}
        except queue.Full:
            return {"ok": False, "error": "MiniCPM local queue is full"}

    def close(self, join=True):
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._queue.put_nowait(("stop", None))
        except queue.Full:
            pass
        if join and self._thread and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            MiniCPMLoader.load()
            self._ready.set()
            self._event({
                "type": "ready",
                "provider": "minicpm",
                "model": "MiniCPM-o-4_5-int4",
                "input_sample_rate": self.input_sample_rate,
            })
            while not self._stop.is_set():
                kind, value = self._queue.get()
                if kind == "stop":
                    break
                if kind == "audio":
                    self._consume_audio(value)
                elif kind == "response":
                    self._flush_audio()
                    self._generate(value)
        except Exception as exc:
            self._error = exc
            LOGGER.exception("minicpm_session_failed error_type=%s", type(exc).__name__)
            self._ready.set()
            self._event({"type": "error", "message": str(exc)})
        finally:
            self._closed.set()
            self._event({"type": "closed"})

    def _consume_audio(self, audio_bytes):
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        now = time.monotonic()
        if rms >= LOCAL_VAD_THRESHOLD:
            if not self._speech_started:
                self._speech_started = True
                self._event({"type": "speech_started"})
            self._last_voice_at = now
        self._audio.extend(audio_bytes)
        max_bytes = int(LOCAL_MAX_AUDIO_SECONDS * LOCAL_SAMPLE_RATE * 2)
        if len(self._audio) > max_bytes:
            self._audio = self._audio[-max_bytes:]
        if (
            self._speech_started
            and self._last_voice_at is not None
            and now - self._last_voice_at >= LOCAL_SILENCE_SECONDS
        ):
            self._flush_audio()

    def _flush_audio(self):
        if not self._speech_started or not self._audio:
            return
        audio = bytes(self._audio)
        self._audio.clear()
        self._speech_started = False
        self._last_voice_at = None
        self._generate(audio)

    def _generate(self, value):
        is_audio = isinstance(value, (bytes, bytearray))
        prompt = "" if is_audio else str(value or "").strip()
        audio = (
            np.frombuffer(value, dtype="<i2").astype(np.float32) / 32768.0
            if is_audio
            else None
        )
        history_user_text = prompt
        if is_audio:
            audio_rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            audio_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            audio_seconds = float(audio.size / LOCAL_SAMPLE_RATE) if audio.size else 0.0
            LOGGER.info(
                "minicpm45_audio bytes=%d seconds=%.3f rms=%.6f peak=%.6f",
                len(value),
                audio_seconds,
                audio_rms,
                audio_peak,
            )
            try:
                # Reuse the cloud server's local Whisper service for context
                # and diagnostics; MiniCPM still receives the original audio.
                from server.stt_proxy import transcribe_float32

                stt_payload = transcribe_float32(
                    np.asarray(audio, dtype="<f4").tobytes(),
                    sample_rate=LOCAL_SAMPLE_RATE,
                )
                history_user_text = str(
                    stt_payload.get("text")
                    if isinstance(stt_payload, dict)
                    else ""
                ).strip()
            except Exception as stt_error:
                LOGGER.warning(
                    "minicpm45_stt_failed error_type=%s error=%s",
                    type(stt_error).__name__,
                    stt_error,
                )
            if not history_user_text:
                history_user_text = "[audio]"
            LOGGER.info(
                "minicpm45_user_transcript chars=%d text=%s",
                len(history_user_text),
                json.dumps(history_user_text, ensure_ascii=False),
            )
        with self._response_lock:
            if self._response_busy:
                return
            self._response_busy = True
        try:
            model, _ = MiniCPMLoader.load()
            messages = []
            for message in self._history:
                role = message.get("role")
                content = message.get("content", "")
                if isinstance(content, list):
                    model_content = content
                else:
                    model_content = [content]
                messages.append({"role": role, "content": model_content})
            current_content = [audio] if audio is not None else []
            if prompt:
                current_content.append(prompt)
            if not current_content:
                current_content = [" "]
            messages.append({"role": "user", "content": current_content})
            system_messages = [
                message for message in messages
                if message.get("role") == "system"
            ]
            system_text = "\n\n".join(
                str(message.get("content") or "")
                for message in system_messages
            )
            LOGGER.info(
                "minicpm45_input history_items=%d system_messages=%d "
                "system_chars=%d system_tail=%s",
                len(messages),
                len(system_messages),
                len(system_text),
                json.dumps(system_text[-500:], ensure_ascii=False),
            )
            self._event({"type": "assistant_started"})
            result = model.chat(
                image=None,
                msgs=messages,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=True,
                enable_thinking=False,
                use_tts_template=False,
                generate_audio=False,
                omni_mode=False,
                stream=True,
                temperature=0.2,
                top_p=0.8,
                repetition_penalty=1.1,
            )
            text = self._emit_text_stream(result)
            self._history.extend([
                {"role": "user", "content": history_user_text},
                {"role": "assistant", "content": text},
            ])
            self._event({"type": "turn_ready", "user_text": history_user_text})
            self._event({"type": "turn_completed", "user_text": history_user_text})
        except Exception as exc:
            LOGGER.exception("minicpm45_generate_failed error_type=%s", type(exc).__name__)
            self._event({"type": "error", "message": str(exc)})
        finally:
            with self._response_lock:
                self._response_busy = False

    def _emit_text_stream(self, result):
        """Parse MiniCPM's incremental event JSON like the normal chat path."""
        # Reuse the normal DeepSeek event protocol parser. The parser keeps an
        # incomplete JSON object in its buffer and emits each completed object
        # immediately, so realtime output does not wait for the whole turn.
        from server.chat_ai import (
            _consume_deepseek_event_objects,
            _handle_deepseek_event,
        )

        stream_state = {
            "buffer": "",
            "raw_text": "",
            "params_emitted": False,
            "params": None,
            "segments": [],
            "segment_ids": set(),
            "end_emitted": False,
        }

        def emit_params(payload):
            if stream_state["params_emitted"]:
                return
            stream_state["params_emitted"] = True
            self._chunk("params", payload)

        def emit_chunk(flag, payload):
            self._chunk(flag, payload)

        def consume_events():
            stream_state["buffer"] = _consume_deepseek_event_objects(
                stream_state["buffer"],
                lambda event: _handle_deepseek_event(
                    event,
                    stream_state,
                    emit_params,
                    emit_chunk,
                ),
            )

        for piece in result:
            candidate = str(piece or "")
            if not candidate:
                continue
            # MiniCPM may expose internal control tokens in its text stream.
            candidate = re.sub(r"<\|.*?\|>", "", candidate)
            if not candidate:
                continue
            stream_state["raw_text"] += candidate
            stream_state["buffer"] += candidate
            consume_events()

        consume_events()
        raw_text = stream_state["raw_text"].strip()
        LOGGER.info(
            "minicpm45_raw_output chars=%d text=%s",
            len(raw_text),
            json.dumps(raw_text, ensure_ascii=False),
        )

        if not stream_state["params"] and not stream_state["segments"]:
            # Keep an ignored/non-JSON model response usable. This is only a
            # fallback; valid event JSON always follows the path above.
            fallback = {
                "action": "reply",
                "mood": "normal",
                "expression": "normal",
                "heart_beat_seconds": "30",
                "special": "",
            }
            self._chunk("params", fallback)
            if raw_text:
                self._chunk(
                    "segment",
                    {"seq": 1, "jp": raw_text, "cn": raw_text},
                )
            self._chunk("end", "")
        elif not stream_state["end_emitted"]:
            stream_state["end_emitted"] = True
            self._chunk("end", "")

        return raw_text



def minicpm_local_start(
    messages=None,
    on_chunk=None,
    on_event=None,
    **kwargs,
):
    return MiniCPMLocalSession(
        messages=messages,
        on_chunk=on_chunk,
        on_event=on_event,
        **kwargs,
    ).start()
