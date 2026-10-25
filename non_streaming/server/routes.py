from flask import jsonify, send_file, request
from flask_socketio import SocketIO

from server.app import load_catalog, get_pet_by_id, get_pet_zip_path, BASE_DIR
from server.auth import register_auth_routes
from server.stt_proxy import SttProxyError, transcribe_float32


def register_routes(app, socketio: SocketIO):
    # Auth routes
    register_auth_routes(app)

    @app.route("/api/pets")
    def list_pets():
        catalog = load_catalog()
        return jsonify({"pets": catalog, "total": len(catalog)})

    @app.route("/api/pets/<pet_id>")
    def get_pet(pet_id: str):
        pet = get_pet_by_id(pet_id)
        if not pet:
            return jsonify({"error": "Pet not found"}), 404
        return jsonify(pet)

    @app.route("/api/pets/<pet_id>/download")
    def download_pet(pet_id: str):
        zip_path = get_pet_zip_path(pet_id)
        if not zip_path:
            return jsonify({"error": "Pet package not found"}), 404
        return send_file(
            zip_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{pet_id}.zip",
        )

    @app.route("/api/pets/<pet_id>/thumbnail")
    def get_thumbnail(pet_id: str):
        thumb_path = BASE_DIR / "server_pets" / "thumbnails" / f"{pet_id}.png"
        if not thumb_path.exists():
            return jsonify({"error": "Thumbnail not found"}), 404
        return send_file(str(thumb_path), mimetype="image/png")

    @app.post("/api/stt/transcribe")
    def transcribe_audio():
        audio_data = request.get_data(cache=False)
        if not audio_data:
            return jsonify({"error": "Audio data is empty"}), 400

        try:
            sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
            return jsonify(transcribe_float32(audio_data, sample_rate))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid sample rate"}), 400
        except SttProxyError as exc:
            return jsonify({"error": str(exc)}), exc.status_code
