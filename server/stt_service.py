import atexit
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from server.logging_utils import get_logger


STT_SERVER_URL = os.environ.get("STT_SERVER_URL", "http://127.0.0.1:18765/transcribe")
STT_STARTUP_TIMEOUT = float(os.environ.get("STT_STARTUP_TIMEOUT", "120"))
STT_AUTOSTART = os.environ.get("STT_AUTOSTART", "1").strip().lower() not in ("0", "false", "no", "off")
STT_PYTHON_EXEC = os.environ.get("STT_PYTHON_EXEC", sys.executable).strip() or sys.executable

_process = None
_lock = threading.Lock()
LOGGER = get_logger("stt_service")


def _service_address() -> tuple[str, int, str]:
    parsed = urlparse(STT_SERVER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    health_url = parsed._replace(path="/health", params="", query="", fragment="").geturl()
    return host, port, health_url


def is_stt_service_ready() -> bool:
    host, port, _ = _service_address()
    try:
        with socket.create_connection((host, port), timeout=1.5) as sock:
            request = (
                "GET /health HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            sock.sendall(request)
            response = sock.recv(128)
            return response.startswith(b"HTTP/") and b" 200 " in response[:32]
    except OSError:
        return False


def ensure_stt_service() -> bool:
    global _process

    if is_stt_service_ready():
        LOGGER.info("stt_ready url=%s", STT_SERVER_URL)
        return True

    if not STT_AUTOSTART:
        LOGGER.warning("stt_unavailable autostart=disabled")
        return False

    host, port, _ = _service_address()
    if host not in ("127.0.0.1", "localhost", "::1"):
        LOGGER.warning("stt_unavailable remote_url=%s", STT_SERVER_URL)
        return False

    with _lock:
        if is_stt_service_ready():
            return True

        if _process is not None and _process.poll() is None:
            return _wait_until_ready(_process)

        script_path = Path(__file__).resolve().parent / "stt_http_server.py"
        command = [STT_PYTHON_EXEC, str(script_path), "--host", host, "--port", str(port)]
        model_path = os.environ.get("STT_MODEL_PATH", "").strip()
        if model_path:
            command.extend(("--model", model_path))

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        LOGGER.info("stt_starting host=%s port=%d", host, port)
        _process = subprocess.Popen(
            command,
            cwd=str(script_path.parent),
            creationflags=creationflags,
        )
        return _wait_until_ready(_process)


def _wait_until_ready(process: subprocess.Popen) -> bool:
    deadline = time.monotonic() + max(1.0, STT_STARTUP_TIMEOUT)
    while time.monotonic() < deadline:
        if is_stt_service_ready():
            LOGGER.info("stt_started pid=%s", process.pid)
            return True
        if process.poll() is not None:
            LOGGER.error("stt_start_failed exit_code=%s", process.returncode)
            return False
        time.sleep(0.5)

    LOGGER.error("stt_start_timeout timeout_seconds=%.0f", STT_STARTUP_TIMEOUT)
    stop_stt_service()
    return False


def stop_stt_service() -> None:
    global _process

    process = _process
    _process = None
    if process is None or process.poll() is not None:
        return

    LOGGER.info("stt_stopping pid=%s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


atexit.register(stop_stt_service)
