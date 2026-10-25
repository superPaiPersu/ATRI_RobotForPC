import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server.app import create_app
from server.logging_utils import get_logger
from server.stt_service import ensure_stt_service, stop_stt_service

app, socketio = create_app()
LOGGER = get_logger("run")

if __name__ == "__main__":
    if not ensure_stt_service():
        LOGGER.warning("stt_unavailable voice_recognition_requests_will_fail=true")

    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
    finally:
        stop_stt_service()
