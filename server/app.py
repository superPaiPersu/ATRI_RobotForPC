import json
import os
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from server.logging_utils import configure_logging

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "server" / "catalog.json"
PETS_DIR = BASE_DIR / "server_pets"


def create_app() -> tuple[Flask, SocketIO]:
    logger = configure_logging()
    app = Flask(__name__)
    CORS(app)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or __import__("secrets").token_hex(32)

    socketio_debug = os.environ.get("AI_PET_SOCKETIO_LOG", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        async_handlers=True,
        # simple-websocket provides Engine.IO WebSocket transport on Windows
        # while preserving the existing threaded server handlers.
        logger=socketio_debug,
        engineio_logger=socketio_debug,
    )

    # Import routes after app creation to avoid circular imports
    from server.routes import register_routes
    register_routes(app, socketio)

    from server.socket_handlers import register_socket_handlers
    register_socket_handlers(socketio)

    logger.info(
        "server_ready socketio_debug=%s async_mode=%s websocket_driver=simple-websocket",
        socketio_debug,
        socketio.async_mode,
    )

    return app, socketio


def load_catalog() -> list[dict]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pet_by_id(pet_id: str) -> dict | None:
    catalog = load_catalog()
    for pet in catalog:
        if pet["id"] == pet_id:
            return pet
    return None


def get_pet_zip_path(pet_id: str) -> str | None:
    zip_path = PETS_DIR / f"{pet_id}.zip"
    if zip_path.exists():
        return str(zip_path)
    return None
