import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from server.logging_utils import get_logger

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DEFAULT_MODEL_PATH = os.environ.get(
    "STT_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "STT模型" / "whisper-small-local-545steps"),
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
SAMPLE_RATE = 16000
LOGGER = get_logger("stt_http")


class STTEngine:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("stt_loading_processor path=%s", model_path)
        self.processor = WhisperProcessor.from_pretrained(model_path, language="zh", task="transcribe")
        LOGGER.info("stt_loading_model device=%s", self.device)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.model.generation_config.suppress_tokens = []
        self.model.generation_config.begin_suppress_tokens = [220, 50257]
        self.model.generation_config.language = "zh"
        self.model.generation_config.task = "transcribe"
        # New Transformers rejects the legacy forced_decoder_ids that some
        # Whisper checkpoints ship with when language/task are passed to
        # generate(). Use the explicit generation arguments below instead.
        self.model.generation_config.forced_decoder_ids = None
        if hasattr(self.model.config, "forced_decoder_ids"):
            self.model.config.forced_decoder_ids = None
        LOGGER.info("stt_warmup_begin")
        self._warmup()
        LOGGER.info("stt_warmup_done")
        LOGGER.info("stt_model_ready device=%s", self.device)

    def _warmup(self) -> None:
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with torch.no_grad():
            inputs = self.processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            self.model.generate(
                inputs.input_features.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=8,
                num_beams=1,
                do_sample=False,
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE}, got {sample_rate}")

        with torch.no_grad():
            inputs = self.processor(
                audio.astype(np.float32, copy=False),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            pred_ids = self.model.generate(
                inputs.input_features.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                language="zh",
                task="transcribe",
                max_new_tokens=48,
                num_beams=1,
                do_sample=False,
                return_timestamps=False,
            )
        return self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def make_handler(engine: STTEngine):
    class STTRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "device": engine.device})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/transcribe":
                self._send_json(404, {"error": "not found"})
                return

            started_at = time.monotonic()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                sample_rate = int(self.headers.get("X-Sample-Rate", str(SAMPLE_RATE)))
                body = self.rfile.read(length)
                audio = np.frombuffer(body, dtype=np.float32)
                text = engine.transcribe(audio, sample_rate)
                self._send_json(200, {"text": text})
                LOGGER.debug(
                    "stt_inference_done bytes=%d chars=%d elapsed_ms=%.1f",
                    length,
                    len(text),
                    (time.monotonic() - started_at) * 1000,
                )
            except Exception as exc:
                LOGGER.warning(
                    "stt_inference_failed elapsed_ms=%.1f error=%s",
                    (time.monotonic() - started_at) * 1000,
                    exc,
                )
                self._send_json(500, {"error": str(exc)})

        def log_message(self, format, *args) -> None:
            return

    return STTRequestHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    engine = STTEngine(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    LOGGER.info("stt_http_ready host=%s port=%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
