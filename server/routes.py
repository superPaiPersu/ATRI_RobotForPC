from flask import jsonify, send_file, request
from flask_socketio import SocketIO

from server.app import load_catalog, get_pet_by_id, get_pet_zip_path, BASE_DIR
from server.auth import register_auth_routes
from server.auth import get_authenticated_username
from server import harness_stream
from server.chat_ai import translate_text
from server.stt_proxy import SttProxyError, transcribe_float32


def register_routes(app, socketio: SocketIO):
    # Auth routes
    register_auth_routes(app)

    @app.route("/")
    def index():
        return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Pet Platform Test</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f6f7fb;
      color: #1f2937;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    main {
      width: min(760px, 100%);
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(15, 23, 42, .08);
      padding: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }
    p {
      margin: 0 0 18px;
      color: #667085;
      line-height: 1.7;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }
    button, a.button {
      border: 0;
      border-radius: 6px;
      background: #2563eb;
      color: white;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
      text-decoration: none;
    }
    a.button.secondary {
      background: #111827;
    }
    pre {
      min-height: 180px;
      overflow: auto;
      margin: 0;
      padding: 14px;
      border-radius: 6px;
      background: #0f172a;
      color: #d1e7ff;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 12px 0 18px;
      color: #047857;
      font-weight: 600;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #10b981;
    }
  </style>
</head>
<body>
  <main>
    <h1>AI Pet Platform</h1>
    <p>服务器已连接到当前后端。这个页面用于快速检查接口是否可用。</p>
    <div class="status"><span class="dot"></span><span>HTTP 入口正常</span></div>
    <div class="actions">
      <button id="pets">测试 /api/pets</button>
      <button id="socket">测试 Socket.IO 握手</button>
      <a class="button secondary" href="/api/pets">打开宠物列表 JSON</a>
    </div>
    <pre id="output">等待测试...</pre>
  </main>
  <script>
    const output = document.getElementById("output");
    async function show(label, url) {
      output.textContent = label + " 请求中...";
      try {
        const response = await fetch(url);
        const text = await response.text();
        output.textContent = [
          label,
          "status: " + response.status,
          "",
          text.slice(0, 3000)
        ].join("\\n");
      } catch (error) {
        output.textContent = label + " failed: " + error;
      }
    }
    document.getElementById("pets").onclick = () => show("/api/pets", "/api/pets");
    document.getElementById("socket").onclick = () => show("Socket.IO", "/socket.io/?EIO=4&transport=polling");
  </script>
</body>
</html>
""", 200, {"Content-Type": "text/html; charset=utf-8"}

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

    @app.post("/api/harness-stream")
    def receive_harness_stream():
        data = request.get_json(silent=True) or {}
        username, auth_error = get_authenticated_username(data, require_token=True)
        if auth_error:
            return jsonify({"ok": False, "error": auth_error}), 401

        session_id = str(data.get("session_id") or "").strip()
        frame = data.get("frame")
        if not session_id or len(session_id) > 200 or not isinstance(frame, dict):
            return jsonify({"ok": False, "error": "Invalid Harness stream payload"}), 400

        chunk = frame.get("chunk")
        if isinstance(chunk, dict):
            text = chunk.get("text")
            if isinstance(text, str) and len(text) > 1024 * 1024:
                return jsonify({"ok": False, "error": "Harness stream frame is too large"}), 413

        accepted, error = harness_stream.accept_frame(username, session_id, frame)
        if not accepted:
            return jsonify({"ok": False, "error": error}), 409
        return jsonify({"ok": True})

    @app.post("/api/translate")
    def translate():
        data = request.get_json(silent=True) or {}
        text = str(data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        source_lang = str(data.get("source_lang") or "ja").strip()
        target_lang = str(data.get("target_lang") or "zh").strip()
        try:
            return jsonify({
                "translated_text": translate_text(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 502

    @app.post("/api/ja2cn")
    def ja2cn():
        data = request.get_json(silent=True) or {}
        text = str(data.get("text") or data.get("data") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        try:
            return jsonify({
                "translated_text": translate_text(text, source_lang="ja", target_lang="zh")
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 502
