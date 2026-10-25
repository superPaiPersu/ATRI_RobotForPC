"""凭据管理模块 - 用于保存和加载用户登录信息"""

import json
import os
from pathlib import Path

CREDENTIALS_FILE = Path.home() / ".ai_pet_platform" / "credentials.json"


def save_credentials(username: str, token: str, password: str = ""):
    """保存用户凭据到本地文件"""
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "username": username,
        "token": token,
    }
    if password:
        data["password"] = password

    tmp = str(CREDENTIALS_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CREDENTIALS_FILE)


def load_credentials() -> dict | None:
    """加载保存的用户凭据"""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def clear_credentials():
    """清除保存的用户凭据"""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def should_auto_login() -> bool:
    """检查是否应该自动登录"""
    creds = load_credentials()
    if not creds:
        return False
    return bool(creds.get("token") or (creds.get("username") and creds.get("password")))
