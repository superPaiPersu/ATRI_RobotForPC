"""Small microphone-to-microphone example for the OpenAI Realtime API.

Run this file from a terminal after setting OPENAI_API_KEY. The Realtime API
uses server-side VAD, so speaking starts a turn and a short pause ends it.
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
from collections import deque
from typing import Any

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection


SAMPLE_RATE = 24_000
CHANNELS = 1
CHUNK_MS = 20
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_MS // 1000
MODEL = "gpt-realtime"


class PcmPlayer:
    """Play 24 kHz mono PCM16 chunks without blocking the event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_FRAMES,
            callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    def add(self, pcm16: bytes) -> None:
        samples = np.frombuffer(pcm16, dtype=np.int16).copy()
        if samples.size == 0:
            return
        with self._lock:
            self._chunks.append(samples)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
        with self._lock:
            output = np.zeros(frames, dtype=np.int16)
            written = 0
            while written < frames and self._chunks:
                chunk = self._chunks.popleft()
                count = min(frames - written, len(chunk))
                output[written : written + count] = chunk[:count]
                written += count
                if count < len(chunk):
                    self._chunks.appendleft(chunk[count:])
            outdata[:, 0] = output


async def send_microphone_audio(connection: AsyncRealtimeConnection) -> None:
    """Continuously append 20 ms PCM16 microphone chunks."""

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_FRAMES,
    ) as stream:
        while True:
            if stream.read_available < CHUNK_FRAMES:
                await asyncio.sleep(0.005)
                continue

            data, _ = stream.read(CHUNK_FRAMES)
            await connection.input_audio_buffer.append(
                audio=base64.b64encode(data.tobytes()).decode("ascii")
            )


async def receive_model_events(connection: AsyncRealtimeConnection, player: PcmPlayer) -> None:
    active_response_id: str | None = None

    async for event in connection:
        event_type = event.type

        if event_type == "input_audio_buffer.speech_started":
            # The server also interrupts its response; discard audio already queued locally.
            player.clear()
            print("\n你：正在说话...", flush=True)
        elif event_type == "response.output_audio.delta":
            player.add(base64.b64decode(event.delta))
        elif event_type == "response.output_audio_transcript.delta":
            if event.response_id != active_response_id:
                active_response_id = event.response_id
                print("模型：", end="", flush=True)
            print(event.delta, end="", flush=True)
        elif event_type == "response.done":
            if active_response_id is not None:
                print()
                active_response_id = None
        elif event_type == "error":
            message = getattr(event, "error", event)
            raise RuntimeError(f"Realtime API error: {message}")


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("请先设置 OPENAI_API_KEY 环境变量。")

    player = PcmPlayer()
    player.start()

    client = AsyncOpenAI()
    try:
        async with client.realtime.connect(model=MODEL) as connection:
            await connection.session.update(
                session={
                    "type": "realtime",
                    "model": MODEL,
                    "output_modalities": ["audio"],
                    "instructions": "用中文自然、简短地和用户进行语音对话。",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                                "create_response": True,
                                "interrupt_response": True,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            "voice": "marin",
                        },
                    },
                }
            )

            print(f"已连接 {MODEL}。开始说话，按 Ctrl+C 结束。")
            await asyncio.gather(
                send_microphone_audio(connection),
                receive_model_events(connection, player),
            )
    finally:
        player.close()
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已结束。")
