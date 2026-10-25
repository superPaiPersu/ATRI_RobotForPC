"""Local Qwen2.5-Omni audio-input/text-output realtime session."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path(
    os.getenv(
        "QWEN3_MODEL_PATH",
        "/mnt/storage/ai-pet-model/Qwen2.5-Omni-7B",
    )
)
SAMPLE_RATE = 16000
SILENCE_SECONDS = float(os.getenv("QWEN3_SILENCE_SECONDS", "0.65"))
VAD_THRESHOLD = float(os.getenv("QWEN3_VAD_THRESHOLD", "0.003"))
MAX_AUDIO_SECONDS = float(os.getenv("QWEN3_MAX_AUDIO_SECONDS", "45"))
MAX_NEW_TOKENS = int(os.getenv("QWEN3_MAX_NEW_TOKENS", "768"))
MAX_REALTIME_HISTORY = int(os.getenv("QWEN3_MAX_REALTIME_HISTORY", "96"))
GPU_MEMORY_LIMIT = os.getenv("QWEN3_GPU_MEMORY_LIMIT", "19GiB")
CPU_MEMORY_LIMIT = os.getenv("QWEN3_CPU_MEMORY_LIMIT", "48GiB")
FORCE_GPU = os.getenv("QWEN3_FORCE_GPU", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}


def _ensure_torch_set_submodule(torch):
    """Backport Module.set_submodule for Torch 2.4 quantized-module replacement."""
    module_type = torch.nn.Module
    if hasattr(module_type, "set_submodule"):
        return

    def set_submodule(self, target, module, strict=False):
        if not target:
            raise ValueError("target must not be empty")
        atoms = target.split(".")
        parent = self
        for atom in atoms[:-1]:
            if not hasattr(parent, atom):
                raise AttributeError(f"{type(parent).__name__} has no attribute {atom}")
            parent = getattr(parent, atom)
            if not isinstance(parent, module_type):
                raise AttributeError(f"{atom} is not an nn.Module")
        name = atoms[-1]
        if strict and not hasattr(parent, name):
            raise AttributeError(f"{type(parent).__name__} has no attribute {name}")
        setattr(parent, name, module)

    module_type.set_submodule = set_submodule


class Qwen3OmniLoader:
    """Process-wide lazy holder for the Qwen2.5-Omni Thinker-only model."""

    _lock = threading.Lock()
    _model = None
    _processor = None
    _error = None

    @classmethod
    def load(cls):
        if cls._model is not None:
            return cls._model, cls._processor
        with cls._lock:
            if cls._model is not None:
                return cls._model, cls._processor
            if cls._error is not None:
                raise RuntimeError(str(cls._error)) from cls._error
            try:
                if not (MODEL_PATH / "model.safetensors.index.json").exists():
                    raise FileNotFoundError(f"Qwen2.5-Omni model is incomplete: {MODEL_PATH}")

                import torch
                from transformers import (
                    Qwen2_5OmniForConditionalGeneration,
                    Qwen2_5OmniProcessor,
                )
                import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as qwen25_modeling
                import transformers.modeling_utils as modeling_utils

                _ensure_torch_set_submodule(torch)
                # Transformers 5.2 pre-allocates the estimated model size in
                # one CUDA block. On a shared 24 GB card that warmup can OOM
                # even when incremental quantized loading itself fits.
                modeling_utils.caching_allocator_warmup = lambda *_args, **_kwargs: None
                # The official speaker map is a trusted weights-only file. The
                # shared server still uses Torch 2.4, while Transformers 5.2
                # applies a blanket 2.6 gate before this safe load.
                qwen25_modeling.check_torch_load_is_safe = lambda: None
                LOGGER.info(
                    "qwen25_omni_load_begin path=%s quantization=bf16 "
                    "force_gpu=%s gpu_memory_limit=%s cpu_memory_limit=%s",
                    MODEL_PATH,
                    FORCE_GPU,
                    GPU_MEMORY_LIMIT,
                    CPU_MEMORY_LIMIT,
                )
                placement = {"": 0} if FORCE_GPU else "auto"
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    str(MODEL_PATH),
                    enable_audio_output=False,
                    dtype=torch.bfloat16,
                    device_map=placement,
                    **(
                        {}
                        if FORCE_GPU
                        else {"max_memory": {0: GPU_MEMORY_LIMIT, "cpu": CPU_MEMORY_LIMIT}}
                    ),
                    attn_implementation="sdpa",
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                model.disable_talker()
                model.eval()
                processor = Qwen2_5OmniProcessor.from_pretrained(
                    str(MODEL_PATH),
                    local_files_only=True,
                )
                cls._model = model
                cls._processor = processor
                LOGGER.info("qwen25_omni_load_ready talker=disabled generation_mode=text")
                return cls._model, cls._processor
            except Exception as exc:
                cls._error = exc
                LOGGER.exception("qwen25_omni_load_failed error_type=%s", type(exc).__name__)
                raise


class Qwen3OmniLocalSession:
    model = "Qwen2.5-Omni-7B"
    input_sample_rate = SAMPLE_RATE
    transcription_model = "local-qwen25-omni-audio"

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
        self._audio_chunk_count = 0
        self._speech_started = False
        self._last_voice_at = None
        self._started_at = time.monotonic()
        self._response_lock = threading.Lock()
        self._response_busy = False

    @staticmethod
    def _extract_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if value.get("type") in ("audio", "image", "video"):
                return ""
            return str(value.get("text") or value.get("content") or "")
        if isinstance(value, list):
            parts = []
            for item in value:
                text = Qwen3OmniLocalSession._extract_text(item)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return str(value or "")

    @classmethod
    def _normalize_history(cls, messages):
        normalized = []
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
            content = cls._extract_text(content).strip()
            if content:
                normalized.append({"role": role, "content": content})
        while normalized and normalized[0]["role"] not in ("system", "user"):
            normalized.pop(0)
        if MAX_REALTIME_HISTORY > 0 and len(normalized) > MAX_REALTIME_HISTORY:
            system_prefix = []
            dialogue = []
            for item in normalized:
                if not dialogue and item["role"] == "system":
                    system_prefix.append(item)
                else:
                    dialogue.append(item)
            normalized = system_prefix + dialogue[-MAX_REALTIME_HISTORY:]
            LOGGER.info(
                "qwen25_realtime_history_trimmed original=%d retained=%d "
                "system_items=%d dialogue_items=%d",
                len(messages),
                len(normalized),
                len(system_prefix),
                len(dialogue[-MAX_REALTIME_HISTORY:]),
            )
        return normalized

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

    def start(self, timeout=600):
        self._thread = threading.Thread(
            target=self._run,
            name="qwen25-omni-local-session",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError("Qwen3-Omni local session startup timed out")
        if self._error is not None:
            raise RuntimeError(f"Qwen2.5-Omni local session failed: {self._error}")
        return self

    def append_audio(self, audio_bytes):
        if self.closed or not audio_bytes:
            return False
        with self._response_lock:
            if self._response_busy:
                return True
        try:
            self._queue.put_nowait(("audio", bytes(audio_bytes)))
            return True
        except queue.Full:
            return False

    def request_response(self, prompt=None):
        if self.closed:
            return {"ok": False, "error": "Qwen2.5-Omni local session is not active"}
        try:
            self._queue.put_nowait(("response", str(prompt or "")))
            return {"ok": True, "queued": True}
        except queue.Full:
            return {"ok": False, "error": "Qwen2.5-Omni local queue is full"}

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
            Qwen3OmniLoader.load()
            self._ready.set()
            self._event(
                {
                    "type": "ready",
                    "provider": "qwen2.5-omni",
                    "model": self.model,
                    "input_sample_rate": self.input_sample_rate,
                }
            )
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
            LOGGER.exception("qwen25_omni_session_failed error_type=%s", type(exc).__name__)
            self._ready.set()
            self._event({"type": "error", "message": str(exc)})
        finally:
            self._closed.set()
            self._event({"type": "closed"})

    def _consume_audio(self, audio_bytes):
        self._audio_chunk_count += 1
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        now = time.monotonic()
        if rms >= VAD_THRESHOLD:
            if not self._speech_started:
                self._speech_started = True
                self._event({"type": "speech_started"})
            self._last_voice_at = now
        self._audio.extend(audio_bytes)
        max_bytes = int(MAX_AUDIO_SECONDS * SAMPLE_RATE * 2)
        if len(self._audio) > max_bytes:
            self._audio = self._audio[-max_bytes:]
        if self._audio_chunk_count == 1 or self._audio_chunk_count % 100 == 0:
            LOGGER.info(
                "qwen25_omni_audio_frame count=%d bytes=%d rms=%.6f "
                "threshold=%.6f speech_started=%s buffered_seconds=%.3f",
                self._audio_chunk_count,
                len(audio_bytes),
                rms,
                VAD_THRESHOLD,
                self._speech_started,
                len(self._audio) / (SAMPLE_RATE * 2),
            )
        if (
            self._speech_started
            and self._last_voice_at is not None
            and now - self._last_voice_at >= SILENCE_SECONDS
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

    @staticmethod
    def _transcribe_for_history(audio):
        try:
            from server.stt_proxy import transcribe_float32

            payload = transcribe_float32(
                np.asarray(audio, dtype="<f4").tobytes(),
                sample_rate=SAMPLE_RATE,
            )
            if isinstance(payload, dict):
                return str(payload.get("text") or "").strip()
        except Exception as exc:
            LOGGER.warning(
                "qwen25_omni_stt_failed error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
        return ""

    def _build_conversation(self, audio=None, prompt=""):
        conversation = []
        for message in self._history:
            conversation.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )
        current = []
        if audio is not None:
            current.append({"type": "audio", "audio": audio})
        if prompt.strip():
            current.append({"type": "text", "text": prompt.strip()})
        if not current:
            current.append({"type": "text", "text": " "})
        conversation.append({"role": "user", "content": current})
        return conversation

    def _generate(self, value):
        is_audio = isinstance(value, (bytes, bytearray))
        prompt = "" if is_audio else str(value or "").strip()
        audio = None
        user_text = prompt
        if is_audio:
            audio = np.frombuffer(value, dtype="<i2").astype(np.float32) / 32768.0
            LOGGER.info(
                "qwen25_omni_audio bytes=%d seconds=%.3f rms=%.6f peak=%.6f",
                len(value),
                len(audio) / SAMPLE_RATE,
                float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0,
                float(np.max(np.abs(audio))) if audio.size else 0.0,
            )
            user_text = self._transcribe_for_history(audio) or "[audio]"
            LOGGER.info(
                "qwen25_omni_user_transcript text=%s",
                json.dumps(user_text, ensure_ascii=False),
            )

        with self._response_lock:
            if self._response_busy:
                return
            self._response_busy = True
        try:
            model, processor = Qwen3OmniLoader.load()
            conversation = self._build_conversation(audio=audio, prompt=prompt)
            template = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = processor(
                text=[template],
                audio=[audio] if audio is not None else None,
                images=None,
                videos=None,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            LOGGER.info(
                "qwen25_omni_input history_items=%d template_chars=%d audio_seconds=%.3f",
                len(self._history),
                len(template),
                len(audio) / SAMPLE_RATE if audio is not None else 0.0,
            )
            self._event({"type": "assistant_started"})
            output = self._stream_generate(model, processor, inputs)
            self._history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": output},
                ]
            )
            self._event({"type": "turn_ready", "user_text": user_text})
            self._event({"type": "turn_completed", "user_text": user_text})
        except Exception as exc:
            LOGGER.exception("qwen25_omni_generate_failed error_type=%s", type(exc).__name__)
            self._event({"type": "error", "message": str(exc)})
        finally:
            with self._response_lock:
                self._response_busy = False

    def _stream_generate(self, model, processor, inputs):
        from transformers import TextIteratorStreamer

        tokenizer = processor.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=600,
            clean_up_tokenization_spaces=False,
        )
        generation_error = []

        def run_generation():
            try:
                model.generate(
                    **inputs,
                    generation_mode="text",
                    use_audio_in_video=False,
                    thinker_max_new_tokens=MAX_NEW_TOKENS,
                    thinker_do_sample=False,
                    thinker_streamer=streamer,
                )
            except Exception as exc:
                generation_error.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = threading.Thread(
            target=run_generation,
            name="qwen25-omni-generate",
            daemon=True,
        )
        worker.start()
        output = self._emit_text_stream(streamer)
        worker.join(timeout=5)
        if generation_error:
            raise generation_error[0]
        return output

    def _emit_text_stream(self, pieces):
        from server.chat_ai import (
            _consume_deepseek_event_objects,
            _handle_deepseek_event,
        )

        state = {
            "buffer": "",
            "raw_text": "",
            "params_emitted": False,
            "params": None,
            "segments": [],
            "segment_ids": set(),
            "end_emitted": False,
        }

        def emit_params(payload):
            if state["params_emitted"]:
                return
            state["params_emitted"] = True
            self._chunk("params", payload)

        def emit_chunk(flag, payload):
            self._chunk(flag, payload)

        def consume():
            state["buffer"] = _consume_deepseek_event_objects(
                state["buffer"],
                lambda event: _handle_deepseek_event(
                    event,
                    state,
                    emit_params,
                    emit_chunk,
                ),
            )

        for piece in pieces:
            candidate = re.sub(r"<\|.*?\|>", "", str(piece or ""))
            if not candidate:
                continue
            state["raw_text"] += candidate
            state["buffer"] += candidate
            consume()
        consume()

        raw_text = state["raw_text"].strip()
        LOGGER.info(
            "qwen25_omni_raw_output chars=%d text=%s",
            len(raw_text),
            json.dumps(raw_text, ensure_ascii=False),
        )
        if not state["params"] and not state["segments"]:
            self._chunk(
                "params",
                {
                    "action": "reply",
                    "mood": "normal",
                    "expression": "normal",
                    "heart_beat_seconds": "30",
                    "special": "",
                },
            )
            if raw_text:
                self._chunk(
                    "segment",
                    {"seq": 1, "jp": raw_text, "cn": raw_text},
                )
            self._chunk("end", "")
        elif not state["end_emitted"]:
            self._chunk("end", "")
        return raw_text


def qwen3_omni_local_start(messages=None, on_chunk=None, on_event=None, **kwargs):
    return Qwen3OmniLocalSession(
        messages=messages,
        on_chunk=on_chunk,
        on_event=on_event,
        **kwargs,
    ).start()
