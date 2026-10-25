from __future__ import annotations

import base64
import ctypes
import io
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image
from PySide6.QtWidgets import QMainWindow, QTabWidget

from api_client import ApiClient
from chat_panel import ChatPanel
from file_access import FileAccessService
from pet_manager import PetManager
from pet_store import PetStoreTab
from socket_client import SocketClient


_IS_WINDOWS = os.name == "nt" and hasattr(ctypes, "windll")


class MainWindow(QMainWindow):
    def __init__(self, username: str, api: ApiClient, socket: SocketClient):
        super().__init__()
        self.username = username
        self.api = api
        self.socket = socket
        self.base_dir = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
        self.pet_process: subprocess.Popen | None = None

        data_dir = self.base_dir / "client_data"
        (data_dir / "pets").mkdir(parents=True, exist_ok=True)
        self.manager = PetManager(str(data_dir))
        self.file_access = FileAccessService(self.base_dir)

        self.setWindowTitle(f"AI桌宠开放平台 — {username}")
        self.resize(820, 680)
        self.setStyleSheet("QMainWindow { background: #1e1e2e; }")
        self._register_screen_capture_handler()
        self._register_file_access_handler()
        self._build_ui()

    def _register_screen_capture_handler(self):
        @self.socket.on("capture_screen")
        def handle_capture_screen(data):
            try:
                payload = data if isinstance(data, dict) else {}
                max_width = int(payload.get("max_width", 720))
                jpeg_quality = int(payload.get("jpeg_quality", 58))
                capture_started_at = datetime.now().isoformat(timespec="milliseconds")
                capture_started_monotonic = time.monotonic()
                idle_seconds = self._idle_seconds()
                window = self._foreground_window_info()
                image = self._capture_screen(max_width)
                image_data = self._jpeg_bytes(image, jpeg_quality)
                capture_finished_monotonic = time.monotonic()
                return {
                    "capture_source": "client",
                    "capture_started_at": capture_started_at,
                    "capture_finished_at": datetime.now().isoformat(timespec="milliseconds"),
                    "capture_latency_seconds": round(capture_finished_monotonic - capture_started_monotonic, 3),
                    "image_width": image.width,
                    "image_height": image.height,
                    "image_base64": base64.b64encode(image_data).decode("ascii"),
                    "idle_seconds": idle_seconds,
                    "window": window,
                }
            except Exception as exc:
                return {"error": str(exc)}

    def _register_file_access_handler(self):
        @self.socket.on("file_request")
        def handle_file_request(data):
            payload = data if isinstance(data, dict) else {}
            return self.file_access.handle_request(payload)

    def _capture_screen(self, max_width: int) -> Image.Image:
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            image = Image.frombytes("RGB", raw.size, raw.rgb)

        if max_width > 0 and image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
        return image

    def _jpeg_bytes(self, image: Image.Image, quality: int) -> bytes:
        quality = max(30, min(95, quality))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()

    def _idle_seconds(self) -> float:
        if not _IS_WINDOWS:
            return 0.0

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("dwTime", ctypes.c_uint),
            ]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
        return max(0.0, millis / 1000.0)

    def _foreground_window_info(self) -> dict[str, str]:
        if not _IS_WINDOWS:
            return {"title": "", "process_name": "", "process_path": ""}

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        MAX_PATH_CHARS = 32768

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return {"title": "", "process_name": "", "process_path": ""}

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value

        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        process_path = ""
        if pid.value:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if handle:
                try:
                    size = ctypes.c_ulong(MAX_PATH_CHARS)
                    path_buffer = ctypes.create_unicode_buffer(MAX_PATH_CHARS)
                    ok = kernel32.QueryFullProcessImageNameW(handle, 0, path_buffer, ctypes.byref(size))
                    if ok:
                        process_path = path_buffer.value
                finally:
                    kernel32.CloseHandle(handle)

        return {
            "title": title,
            "process_name": Path(process_path).name if process_path else "",
            "process_path": process_path,
        }

    def _build_ui(self):
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setStyleSheet("""
            QTabBar::tab {
                background: #313244; color: #a6adc8; padding: 8px 24px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #1e1e2e; color: #cdd6f4; }
            QTabBar::tab:hover { background: #45475a; }
        """)
        self.setCentralWidget(tabs)

        store_tab = PetStoreTab(self.api, self.manager)
        store_tab.pet_activated.connect(self._on_pet_activated)
        tabs.addTab(store_tab, "宠物商店")

        chat_tab = ChatPanel(self.socket, self.manager)
        tabs.addTab(chat_tab, "聊天")
        chat_tab.connect_socket()
        if self.socket.connected:
            payload = {"client_type": "pc", "screen_capture": True, "file_access": True}
            if getattr(self.api, "token", None):
                payload["token"] = self.api.token
            try:
                result = self.socket.call("register_screen_source", payload, timeout=10)
                if isinstance(result, dict) and result.get("error"):
                    chat_tab.chat_log.append(f'<span style="color: #f38ba8;">PC 能力注册失败: {result["error"]}</span>')
                    print("register_screen_source failed:", result)
            except Exception as exc:
                chat_tab.chat_log.append(f'<span style="color: #f38ba8;">PC 能力注册异常: {exc}</span>')
                print("register_screen_source exception:", exc)

        store_tab.load_catalog()

    def _on_pet_activated(self, pet_id: str):
        self._stop_pet_process()
        if not pet_id:
            return

        installed = self.manager.list_installed()
        if pet_id not in installed:
            return

        template_main = self.base_dir / "pet_template" / "main.py"
        if not template_main.exists():
            print(f"错误: 模板入口不存在 {template_main}")
            return

        cmd = [
            sys.executable,
            str(template_main),
            "--pet-dir", installed[pet_id]["local_path"],
            "--server-url", self.socket.base_url,
            "--username", self.socket.username,
            "--pet-id", pet_id,
        ]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.pet_process = subprocess.Popen(
            cmd,
            cwd=str(template_main.parent),
            creationflags=flags,
        )
        print(f"已启动桌宠 {pet_id}, PID={self.pet_process.pid}")
        self.socket.select_pet(pet_id)

    def _stop_pet_process(self):
        proc = self.pet_process
        self.pet_process = None
        if not proc or proc.poll() is not None:
            return

        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        except Exception as exc:
            print(f"发送停止信号失败 PID={proc.pid}: {exc}")
            try:
                proc.terminate()
            except Exception:
                pass

        threading.Thread(
            target=self._wait_stopped_pet_process,
            args=(proc,),
            daemon=True,
        ).start()

    def _wait_stopped_pet_process(self, proc: subprocess.Popen):
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"pet process stop timed out, killing PID={proc.pid}")
            proc.kill()
            proc.wait(timeout=5)
        print(f"stopped pet process PID={proc.pid}")

    def closeEvent(self, event):
        self._stop_pet_process()
        self.socket.disconnect()
        super().closeEvent(event)
