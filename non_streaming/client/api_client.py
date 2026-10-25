from __future__ import annotations

import requests


class ApiClient:
    """HTTP 客户端封装，与服务端 REST API 通信"""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self.username: str | None = None

    def _headers(self) -> dict:
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def is_reachable(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/pets", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_pets(self) -> list[dict]:
        r = requests.get(f"{self.base_url}/api/pets", timeout=10, headers=self._headers())
        r.raise_for_status()
        return r.json().get("pets", [])

    def get_pet(self, pet_id: str) -> dict:
        r = requests.get(f"{self.base_url}/api/pets/{pet_id}", timeout=10, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def get_download_url(self, pet_id: str) -> str:
        return f"{self.base_url}/api/pets/{pet_id}/download"

    # --- Auth ---

    def register(self, username: str, password: str) -> dict:
        r = requests.post(
            f"{self.base_url}/api/register",
            json={"username": username, "password": password},
            timeout=10,
        )
        if r.status_code != 201:
            raise RuntimeError(r.json().get("error", "注册失败"))
        data = r.json()
        self.username = data["username"]
        self.token = data["token"]
        return data

    def login(self, username: str, password: str) -> dict:
        r = requests.post(
            f"{self.base_url}/api/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if r.status_code != 200:
            raise RuntimeError(r.json().get("error", "登录失败"))
        data = r.json()
        self.username = data["username"]
        self.token = data["token"]
        return data

    def verify_token(self, username: str, token: str) -> bool:
        try:
            r = requests.post(
                f"{self.base_url}/api/verify",
                json={"username": username, "token": token},
                timeout=5,
            )
            return r.status_code == 200 and r.json().get("valid", False)
        except Exception:
            return False

    def refresh_token(self, username: str, token: str) -> dict:
        """刷新 token"""
        r = requests.post(
            f"{self.base_url}/api/refresh",
            json={"username": username, "token": token},
            timeout=10,
        )
        if r.status_code != 200:
            raise RuntimeError(r.json().get("error", "刷新失败"))
        data = r.json()
        self.username = data["username"]
        self.token = data["token"]
        return data
