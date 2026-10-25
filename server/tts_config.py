"""Server-side TTS selection. Edit these variables and restart the server."""

# Keep the existing backend unless explicitly selected here or in a request.
# Options: "new_api" (existing trained voices), "api_v2" (reference audio).
TTS_BACKEND = "new_api"
API_V2_URL = "http://127.0.0.1:9881/tts"
NEW_API_URL = "http://127.0.0.1:9870/tts"
# Matches the local api_v2 process; change this port for another deployment.

NEW_API_TIMEOUT_SECONDS = 60
API_V2_TIMEOUT_SECONDS = 120

