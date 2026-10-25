from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from server import chat_ai


ChunkCallback = Callable[[str, object], None]


@dataclass
class HarnessStreamRequest:
    """One in-flight Harness turn receiving frames from the desktop runtime."""

    username: str
    session_id: str
    messages: object
    on_chunk: ChunkCallback
    stream_state: dict = field(default_factory=chat_ai.new_harness_stream_state)
    lock: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )
    attempt_id: str = ""
    attempt_ended: bool = False
    last_frame_at: float = field(default_factory=time.monotonic)

    def accept(self, frame: dict) -> bool:
        if not isinstance(frame, dict):
            return False

        frame_type = str(frame.get("type") or "").strip().lower()
        attempt_id = str(
            frame.get("attempt_id") or frame.get("attemptId") or ""
        ).strip()

        with self.lock:
            self.last_frame_at = time.monotonic()
            if frame_type == "start":
                self._start_attempt(attempt_id)
                self.lock.notify_all()
                return True

            if attempt_id and self.attempt_id and attempt_id != self.attempt_id:
                return False

            if frame_type == "chunk":
                chunk = frame.get("chunk")
                if not isinstance(chunk, dict) or chunk.get("type") != "text-delta":
                    return False
                text = chunk.get("text")
                if not isinstance(text, str) or not text:
                    return False
                if not chat_ai.is_summary_turn(self.messages):
                    chat_ai.consume_harness_text_delta(
                        text,
                        self.stream_state,
                        on_chunk=self.on_chunk,
                    )
                self.lock.notify_all()
                return True

            if frame_type == "end":
                self.attempt_ended = True
                self.lock.notify_all()
                return True

        return False

    def _start_attempt(self, attempt_id: str) -> None:
        # Harness may perform several model attempts while executing tools or
        # retrying. Only the current attempt can be spoken by the pet.
        if self.attempt_id and attempt_id != self.attempt_id:
            self.stream_state = chat_ai.new_harness_stream_state()
        self.attempt_id = attempt_id
        self.attempt_ended = False

    def wait_for_quiet(self, timeout: float, quiet_seconds: float = 0.2) -> bool:
        """Wait until queued plugin callbacks have stopped changing the state."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.lock:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                quiet_remaining = quiet_seconds - (time.monotonic() - self.last_frame_at)
                if quiet_remaining <= 0:
                    return True
                self.lock.wait(min(remaining, quiet_remaining))

    def finish(self, content) -> dict:
        with self.lock:
            return chat_ai.finish_harness_stream(
                content,
                messages=self.messages,
                on_chunk=self.on_chunk,
                stream_state=self.stream_state,
            )

    def has_emitted_reply(self) -> bool:
        with self.lock:
            return bool(
                self.stream_state.get("params_emitted")
                or self.stream_state.get("segments")
            )


_REGISTRY: dict[str, HarnessStreamRequest] = {}
_REGISTRY_LOCK = threading.RLock()


def register(
    username: str,
    session_id: str,
    messages,
    on_chunk: ChunkCallback,
) -> HarnessStreamRequest:
    username = str(username or "").strip()
    session_id = str(session_id or "").strip()
    if not username or not session_id:
        raise ValueError("Harness stream identity is incomplete")

    request = HarnessStreamRequest(
        username=username,
        session_id=session_id,
        messages=messages,
        on_chunk=on_chunk,
    )
    with _REGISTRY_LOCK:
        if session_id in _REGISTRY:
            raise RuntimeError(f"Harness session already has an active request: {session_id}")
        _REGISTRY[session_id] = request
    return request


def unregister(request: HarnessStreamRequest) -> None:
    with _REGISTRY_LOCK:
        if _REGISTRY.get(request.session_id) is request:
            _REGISTRY.pop(request.session_id, None)


def accept_frame(username: str, session_id: str, frame: dict) -> tuple[bool, str]:
    username = str(username or "").strip()
    session_id = str(session_id or "").strip()
    with _REGISTRY_LOCK:
        active = _REGISTRY.get(session_id)

    if active is None:
        return False, "Harness stream request is not active"
    if active.username != username:
        return False, "Harness stream identity does not match"
    if not active.accept(frame):
        return False, "Unsupported Harness stream frame"
    return True, ""
