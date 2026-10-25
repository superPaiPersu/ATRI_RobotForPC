from __future__ import annotations

import socketio


class SocketClient:
    """WebSocket 客户端封装"""

    def __init__(self, base_url: str = "http://127.0.0.1:5000", username: str = ""):
        self.base_url = base_url
        self.username = username
        self.sio = socketio.Client()
        self._callbacks: dict[str, list] = {}

    def connect(self) -> bool:
        try:
            self.sio.connect(self.base_url)
            return True
        except Exception:
            return False

    def disconnect(self):
        if self.sio.connected:
            self.sio.disconnect()

    @property
    def connected(self) -> bool:
        return self.sio.connected

    def on(self, event: str, callback=None):
        """注册事件回调，支持装饰器和直接调用"""
        if callback is not None:
            print(333)
            self.sio.on(event)(callback)
            return callback
        # Decorator mode: return a decorator
        def decorator(fn):
            self.sio.on(event)(fn)
            return fn
        return decorator

    def emit(self, event: str, data: dict):
        """发送事件，自动注入 username"""
        if self.sio.connected:
            data["username"] = self.username
            self.sio.emit(event, data)

    def call(self, event: str, data: dict, timeout: int = 10):
        if not self.sio.connected:
            raise RuntimeError("socket is not connected")
        data["username"] = self.username
        return self.sio.call(event, data, timeout=timeout)

    def select_pet(self, pet_id: str):
        self.emit("select_pet", {"pet_id": pet_id})

    def send_chat(self, msg: str):
        self.emit("chat", {"msg": msg})

    def link_netease_music(self, pet_id: str):
        self.emit("link_netease_music", {"pet_id": pet_id})

    def check_netease_cookie(self, cellphone_num: str):
        return self.call("netease_check_cookie", {
            "cellphone_num": cellphone_num,
        })

    def request_netease_code(self, pet_id: str, cellphone_num: str):
        self.emit("netease_request_code", {
            "pet_id": pet_id,
            "cellphone_num": cellphone_num,
        })

    def submit_netease_code(self, pet_id: str, cellphone_num: str, code: str):
        self.emit("netease_verify", {
            "pet_id": pet_id,
            "cellphone_num": cellphone_num,
            "code": code,
        })
