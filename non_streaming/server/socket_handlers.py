import base64
import io
import json
import os.path
import pickle
import threading
from datetime import datetime
from pathlib import Path

import requests
import numpy as np
from pydub import AudioSegment
from flask_socketio import emit
from flask import request

from server.auth import get_authenticated_username
from server import ncm
from server.stt_proxy import SttProxyError, transcribe_float32
from server.screen_skill import reset_screen_capture_request, set_screen_capture_request
from server.file_skill import reset_file_access_request, set_file_access_request
from server import task_agent
from server import chat_ai


SERVER_PETS_DIR = Path(__file__).resolve().parent.parent / "server_pets"
DIR_PATH=os.path.dirname(os.path.abspath(__file__))
REQUIRE_SOCKET_AUTH = os.environ.get("AI_PET_REQUIRE_SOCKET_AUTH", "").strip().lower() in ("1", "true", "yes", "on")
_SCREEN_SOURCE_SIDS: dict[str, str] = {}
_CLIENT_CAPABILITIES: dict[str, dict] = {}
_SCREEN_SOURCE_LOCK = threading.Lock()
FILE_TOOL_NAMES = {
    "list_authorized_dirs",
    "list_user_files",
    "search_user_files",
    "read_user_file",
}


SCREEN_TOOL_NAMES = {"inspect_screen_activity"}


def _authenticated_username(data):
    return get_authenticated_username(data if isinstance(data, dict) else {}, require_token=REQUIRE_SOCKET_AUTH)


def _auth_error_dict(message):
    return {"error": message or "认证失败"}


def _auth_error_status(message):
    return "failed----" + (message or "认证失败")


def _auth_error_list(message):
    return ["failed", message or "认证失败"]


def _history_file_path(username, pet_id, filename):
    safe_username = Path(str(username or "unknown_user")).name
    safe_pet_id = Path(str(pet_id or "default")).name
    history_dir = Path(DIR_PATH) / "users_data" / safe_username / safe_pet_id
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / filename


def _convert_to_16_bit_wav(data):
    data = data.copy()
    dtype_name = str(data.dtype)
    if dtype_name in ("float64", "float32", "float16"):
        peak = abs(data).max()
        if peak > 0:
            data = data / peak
        data = data * 32767
        data = data.astype("int16")
    elif dtype_name == "int32":
        data = (data / 65536).astype("int16")
    elif dtype_name == "int16":
        pass
    elif dtype_name == "uint16":
        data = (data - 32768).astype("int16")
    elif dtype_name == "uint8":
        data = (data * 257 - 32768).astype("int16")
    elif dtype_name == "int8":
        data = (data * 256).astype("int16")
    else:
        raise ValueError(f"Audio data cannot be converted from {data.dtype} to int16.")

    return data


def _audio_array_to_wav_bytes(sample_rate, data):
    data = _convert_to_16_bit_wav(data)
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=(1 if len(data.shape) == 1 else data.shape[1]),
    )
    output = io.BytesIO()
    audio.export(output, format="wav")
    return output.getvalue()


def _normalize_tts_response_to_wav_bytes(content):
    if content[:4] == b"RIFF":
        return content

    try:
        data = pickle.loads(content)
    except Exception:
        return content

    return _audio_array_to_wav_bytes(data[-1][0], data[-1][1])


def register_socket_handlers(socketio):
    """Register websocket event handlers."""

    @socketio.on("register_screen_source")
    def register_screen_source(data):
        username, auth_error = _authenticated_username(data)
        if auth_error:
            return _auth_error_dict(auth_error)

        payload = data if isinstance(data, dict) else {}
        if payload.get("screen_capture") is False and payload.get("file_access") is False:
            return {"ok": True, "registered": False}

        sid = getattr(request, "sid", None)
        if not sid:
            return {"error": "Missing socket sid"}

        with _SCREEN_SOURCE_LOCK:
            _SCREEN_SOURCE_SIDS[username] = sid
            _CLIENT_CAPABILITIES[username] = {
                "client_type": payload.get("client_type") or "",
                "screen_capture": bool(payload.get("screen_capture")),
                "file_access": bool(payload.get("file_access")),
            }
        return {"ok": True}

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = getattr(request, "sid", None)
        if not sid:
            return

        with _SCREEN_SOURCE_LOCK:
            stale_users = [username for username, stored_sid in _SCREEN_SOURCE_SIDS.items() if stored_sid == sid]
            for username in stale_users:
                _SCREEN_SOURCE_SIDS.pop(username, None)
                _CLIENT_CAPABILITIES.pop(username, None)

    @socketio.on("select_pet")
    def handle_select_pet(data):
        username = data.get("username", "unknown_user")
        pet_id = data.get("pet_id")
        emit("server_msg", {"msg": f"{username} selected pet: {pet_id}"})

    @socketio.on("chat")
    def handle_chat(data):
        username = data.get("username", "unknown_user")
        msg = data.get("msg", "")
        if not msg:
            return
        emit("server_msg", {"msg": f"[{username}] sent: {msg}"})

    @socketio.on("link_netease_music")
    def handle_link_netease_music(data):
        username = data.get("username", "unknown_user")
        pet_id = data.get("pet_id")
        emit("server_msg", {"msg": f"{username}'s {pet_id} NetEase Music linked"})

    @socketio.on("netease_check_cookie")
    def handle_netease_check_cookie(data):
        cellphone_num = data.get("cellphone_num", "")
        cached = ncm.load_cookie(cellphone_num)
        if cached:
            try:
                ncm.user_login_by_cookie(cellphone_num, cached)
                refreshed = ncm.load_cookie(cellphone_num)
                status = ncm.ncm.login_status(cookie=refreshed)
                if status.body.get("data", {}).get("code") == 200:
                    return {
                        "msg": "妫€娴嬪埌缂撳瓨 cookie锛屽凡鑷姩鐧诲綍",
                        "cookie_valid": True,
                        "cookie_return": refreshed,
                    }
            except Exception as e:
                return {"msg": f"cookie 鍒锋柊澶辫触: {e}", "cookie_valid": False}

        return {"msg": "鏈娴嬪埌鏈夋晥 cookie锛岃鎵嬪姩楠岃瘉", "cookie_valid": False}

    @socketio.on("netease_verify")
    def handle_netease_verify(data):
        code = data.get("code", "")
        cellphone_num = data.get("cellphone_num", "")
        result = ncm.user_login_check_code(cellphone_num, code)
        if result.get("status") == "success":
            ncm.save_cookie(cellphone_num, result.get("cookie"))
            emit(
                "server_msg",
                {
                    "msg": "缃戞槗浜戦煶涔愬叧鑱旀垚鍔燂紒",
                    "verification_result": True,
                    "success": True,
                    "cookie_return": result.get("cookie"),
                },
            )
        else:
            emit(
                "server_msg",
                {
                    "msg": f"鍏宠仈澶辫触: {result.get('status')} 閿欒鐮佷负 {result.get('code')}",
                    "verification_result": True,
                    "success": False,
                },
            )

    @socketio.on("netease_request_code")
    def handle_netease_request_code(data):
        cellphone_num = data.get("cellphone_num", "")
        ncm.user_login_send_code(cellphone_num)
        emit("server_msg", {"msg": f"宸插悜鎵嬫満鍙?{cellphone_num} 鍙戦€侀獙璇佺爜", "code_sent": True})

    @socketio.on("get_answer")
    def get_ai_answer(data):
        _, auth_error = _authenticated_username(data)
        if auth_error:
            return _auth_error_dict(auth_error)
        username = (data.get("username") or "").strip()
        with _SCREEN_SOURCE_LOCK:
            screen_sid = _SCREEN_SOURCE_SIDS.get(username)
            capabilities = dict(_CLIENT_CAPABILITIES.get(username) or {})

        pet_id = data.get("pet_id")
        request_sid = request.sid

        def _emit_chat_chunk(flag, payload):
            socketio.emit("chat_chunk", [flag, payload], to=request_sid)

        disabled_tools = set()
        if not capabilities.get("screen_capture"):
            disabled_tools.update(SCREEN_TOOL_NAMES)
        if not capabilities.get("file_access"):
            disabled_tools.update(FILE_TOOL_NAMES)
        has_agent_capability = bool(screen_sid and (capabilities.get("screen_capture") or capabilities.get("file_access")))

        if not has_agent_capability:
            token = chat_ai.set_disabled_tools(disabled_tools)
            try:
                return chat_ai.get_answer(
                    data.get("model_name"),
                    data.get("api_url"),
                    data.get("api_key"),
                    data.get("data"),

                )
            finally:
                chat_ai.reset_disabled_tools(token)

        def _request_client_capture(payload):
            return socketio.call("capture_screen", payload, to=screen_sid, timeout=30)

        def _request_client_file(payload):
            return socketio.call("file_request", payload, to=screen_sid, timeout=45)

        screen_token = set_screen_capture_request(_request_client_capture)
        file_token = set_file_access_request(_request_client_file)
        disabled_token = chat_ai.set_disabled_tools(disabled_tools)
        try:
            return task_agent.run_task_turn(
                data.get("model_name"),
                data.get("api_url"),
                data.get("api_key"),
                data.get("data"),
                username=username,
                pet_id=pet_id,

            )
        finally:
            chat_ai.reset_disabled_tools(disabled_token)
            reset_file_access_request(file_token)
            reset_screen_capture_request(screen_token)

    @socketio.on("audio_translate")
    def audio_translate(data):
        _, auth_error = _authenticated_username(data)
        if auth_error:
            return _auth_error_dict(auth_error)

        payload = data.get("data") if isinstance(data, dict) else None
        if not isinstance(payload, dict):
            return {"error": "Invalid STT payload"}

        try:
            encoded_audio = payload.get("audio", "")
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
            encoding = str(payload.get("encoding", "pcm_s16le")).lower()
            sample_rate = int(payload.get("sample_rate", 16000))

            if encoding == "pcm_s16le":
                if len(audio_bytes) % 2 != 0:
                    return {"error": "PCM16 audio byte count must be even"}
                samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
                audio_bytes = samples.astype("<f4", copy=False).tobytes()
            elif encoding != "float32le":
                return {"error": f"Unsupported STT audio encoding: {encoding}"}

            return transcribe_float32(audio_bytes, sample_rate)
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid STT payload: {exc}"}
        except SttProxyError as exc:
            return {"error": str(exc)}

    @socketio.on("update_summary_history")
    def update_summary_history(data):
        try:
            user_name, auth_error = _authenticated_username(data)
            if auth_error:
                return _auth_error_status(auth_error)

            pet_id = data.get("pet_id")
            ans = get_ai_answer(data)
            if ans:
                ans = ans['content']
                summary = json.loads(ans)
                if summary.get("action") == "no_sense":
                    return "success"
                if summary.get("action") != "summary":
                    return "failed----unsupported summary action"
                rows = [
                    {
                        "type": "summary_session_start",
                        "started_at": data.get("special"),
                        "marker": "鍞ら啋model",
                        "messages": [],
                    },
                    {
                        "type": "summary_message",
                        "messages": [
                            {
                                "role": "user",
                                "content": str(summary.get("user") or "").strip(),
                            },
                        ],
                    },
                    {
                        "type": "summary_message",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": str(summary.get("model") or "").strip(),
                            },
                        ],
                    },
                    {
                        "type": "summary_session_end",
                        "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "messages": [],
                    },
                ]
                txt = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                with open(_history_file_path(user_name, pet_id, "summary_history.txt"), "a", encoding="utf-8") as f:
                    f.write(txt)
            return "success"
        except Exception as e:
            return "failed----" + str(e)
    @socketio.on("save_dialogue_history")
    def save_dialogue_history(data):
        try:
            user_name, auth_error = _authenticated_username(data)
            if auth_error:
                return _auth_error_status(auth_error)

            pet_id = data.get("pet_id")
            content=data.get("data")
            with open(_history_file_path(user_name, pet_id, "chat_history.txt"), "a", encoding="utf-8") as f:
                f.write(content)
            return "success"
        except Exception as e:
            return "failed----"+str(e)

    @socketio.on("get_summary_history")
    def get_summary_history(data):

        user_name, auth_error = _authenticated_username(data)
        if auth_error:
            return _auth_error_list(auth_error)

        pet_id = data.get("pet_id")
        try:
            pt = _history_file_path(user_name, pet_id, "summary_history.txt")
            if pt.exists():
                with open(pt, "r", encoding="utf-8") as f:
                    txt = f.read()
            else:
                with open(pt, "w", encoding="utf-8") as f:
                    f.write("")
                    txt = ""
            return ["success", txt]

        except Exception as e:
            return ["failed", str(e)]
    @socketio.on("get_dialogue_history")
    def get_dialogue_history(data):

        user_name, auth_error = _authenticated_username(data)
        if auth_error:
            return _auth_error_list(auth_error)

        pet_id = data.get("pet_id")
        try:
            pt = _history_file_path(user_name, pet_id, "chat_history.txt")
            if pt.exists():
                with open(pt,"r",encoding="utf-8") as f:
                    txt=f.read()
            else:
                with open(pt, "w", encoding="utf-8") as f:
                    f.write("")
                    txt=""
            return ["success",txt]

        except Exception as e:
            return ["failed",str(e)]


    @socketio.on("voice_produce")
    def voice_produce(data):
        _, auth_error = _authenticated_username(data)
        if auth_error:
            return {
                "ok": False,
                "error": auth_error,
                "detail": "",
            }

        payload = dict(data.get("data", {}))
        pet_id = payload.pop("pet_id", "")
        ref_audio_name = payload.pop("ref_audio_name", "")
        return_format = payload.pop("return_format", "")
        ref_audio_basename = Path(str(ref_audio_name)).name

        if not pet_id:
            return {
                "ok": False,
                "error": "voice_produce failed: missing pet_id",
                "detail": "",
            }

        if not ref_audio_basename:
            return {
                "ok": False,
                "error": "voice_produce failed: missing ref_audio_name",
                "detail": "",
            }
        ref_audio_path = SERVER_PETS_DIR / pet_id / "config/voice_refs" / ref_audio_basename
        if not ref_audio_path.exists():
            return {
                "ok": False,
                "error": "voice_produce failed: reference audio not found",
                "detail": str(ref_audio_path),
            }

        payload["ref_audio_path"] = str(ref_audio_path)
        try:
            response = requests.post(
                url="http://127.0.0.1:9870/tts",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            if return_format == "wav":
                return _normalize_tts_response_to_wav_bytes(response.content)

            return response.content
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                try:
                    detail = exc.response.text[:200]
                except Exception:
                    detail = "<unreadable response body>"
            return {
                "ok": False,
                "error": f"TTS service request failed: {exc}",
                "detail": detail,
            }
