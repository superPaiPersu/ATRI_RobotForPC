import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from flask import jsonify, request

USERS_FILE = Path(__file__).resolve().parent / "data" / "users.json"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{2,20}$")
PASSWORD_MIN_LENGTH = 6
PBKDF2_ITERATIONS = 200_000


def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"users": {}}


def _save_users(data: dict):
    os.makedirs(USERS_FILE.parent, exist_ok=True)
    tmp = str(USERS_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, USERS_FILE)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password_v2(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        PBKDF2_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def _verify_password(password: str, password_hash: str) -> bool:
    password_hash = password_hash or ""
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt, expected = password_hash.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("ascii"),
                int(iterations),
            ).hex()
            return hmac.compare_digest(digest, expected)
        except (TypeError, ValueError):
            return False

    return hmac.compare_digest(password_hash, _hash_password(password))


def _validate_username(username: str) -> str:
    username = (username or "").strip()
    if not username:
        return "username cannot be empty"
    if not USERNAME_RE.fullmatch(username):
        return "username must be 2-20 chars and may contain letters, digits, underscore, hyphen and Chinese characters"
    return ""


def _validate_password(password: str) -> str:
    if not password:
        return "password cannot be empty"
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"password must be at least {PASSWORD_MIN_LENGTH} characters"
    return ""


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def verify_user_token(username: str, token: str) -> bool:
    username = (username or "").strip()
    token = token or ""
    if not username or not token:
        return False

    users = _load_users()
    user = users.get("users", {}).get(username)
    return bool(user and hmac.compare_digest(user.get("token", ""), token))


def get_authenticated_username(payload: dict, require_token: bool = False) -> tuple[str | None, str]:
    if not isinstance(payload, dict):
        return None, "invalid auth payload"

    username = (payload.get("username") or "").strip()
    token = payload.get("token") or payload.get("auth_token") or ""
    if not username:
        return None, "username cannot be empty"

    if token:
        if verify_user_token(username, token):
            return username, ""
        return None, "login expired, please sign in again"

    if require_token:
        return None, "missing login token"

    return username, ""


def register_auth_routes(app):

    @app.route("/api/register", methods=["POST"])
    def register():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        error = _validate_username(username) or _validate_password(password)
        if error:
            return jsonify({"error": error}), 400

        users = _load_users()
        users.setdefault("users", {})
        if username in users["users"]:
            return jsonify({"error": "用户名已存在"}), 409

        token = _new_token()
        now = _now_iso()
        users["users"][username] = {
            "password_hash": _hash_password_v2(password),
            "token": token,
            "created_at": now,
            "updated_at": now,
        }
        _save_users(users)

        return jsonify({"username": username, "token": token}), 201

    @app.route("/api/login", methods=["POST"])
    def login():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400

        users = _load_users()
        user = users.get("users", {}).get(username)
        if not user or not _verify_password(password, user.get("password_hash", "")):
            return jsonify({"error": "用户名或密码错误"}), 401

        token = _new_token()
        users["users"][username]["token"] = token
        users["users"][username]["updated_at"] = _now_iso()
        if not user.get("password_hash", "").startswith("pbkdf2_sha256$"):
            users["users"][username]["password_hash"] = _hash_password_v2(password)
        _save_users(users)

        return jsonify({"username": username, "token": token}), 200

    @app.route("/api/verify", methods=["POST"])
    def verify():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        token = data.get("token") or ""

        if verify_user_token(username, token):
            return jsonify({"valid": True}), 200

        return jsonify({"valid": False}), 401

    @app.route("/api/refresh", methods=["POST"])
    def refresh():
        """通过旧 token 刷新获取新 token。"""
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        token = data.get("token") or ""

        if not username or not token:
            return jsonify({"error": "用户名和 token 不能为空"}), 400

        users = _load_users()
        user = users.get("users", {}).get(username)
        if not user or not hmac.compare_digest(user.get("token", ""), token):
            return jsonify({"error": "无效的 token"}), 401

        new_token = _new_token()
        users["users"][username]["token"] = new_token
        users["users"][username]["updated_at"] = _now_iso()
        _save_users(users)

        return jsonify({"username": username, "token": new_token}), 200

    @app.route("/api/logout", methods=["POST"])
    def logout():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        token = data.get("token") or ""

        users = _load_users()
        user = users.get("users", {}).get(username)
        if not user or not hmac.compare_digest(user.get("token", ""), token):
            return jsonify({"ok": True}), 200

        users["users"][username]["token"] = ""
        users["users"][username]["updated_at"] = _now_iso()
        _save_users(users)
        return jsonify({"ok": True}), 200

