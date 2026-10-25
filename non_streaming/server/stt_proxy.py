import os

import requests


STT_SERVER_URL = os.environ.get("STT_SERVER_URL", "http://127.0.0.1:18765/transcribe")
STT_REQUEST_TIMEOUT = (3.05, 60)


class SttProxyError(RuntimeError):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def transcribe_float32(audio_data, sample_rate=16000):
    if not audio_data:
        raise SttProxyError("Audio data is empty", 400)

    try:
        response = requests.post(
            STT_SERVER_URL,
            data=audio_data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": str(sample_rate),
            },
            timeout=STT_REQUEST_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise SttProxyError("STT server request timed out", 504) from exc
    except requests.RequestException as exc:
        raise SttProxyError(f"STT server request failed: {exc}", 502) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        message = response.text.strip() or "STT server returned invalid JSON"
        raise SttProxyError(message, 502) from exc

    if not response.ok:
        message = payload.get("error", "STT server request failed") if isinstance(payload, dict) else str(payload)
        raise SttProxyError(message, 502)

    if not isinstance(payload, dict):
        raise SttProxyError("STT server returned an invalid payload", 502)

    return payload
