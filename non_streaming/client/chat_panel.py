from __future__ import annotations

import json
import os

from PySide6.QtCore import Qt, Signal, QEventLoop, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QComboBox,
    QInputDialog,
)

from socket_client import SocketClient
from pet_manager import PetManager
from netease_dialog import NeteaseDialog


class ChatPanel(QWidget):
    """聊天标签页：桌宠选择 + 对话界面。"""

    netease_verified = Signal(bool)

    def __init__(self, socket_client: SocketClient, pet_manager: PetManager, parent=None):
        super().__init__(parent)
        self.socket = socket_client
        self.manager = pet_manager
        self._setup_ui()
        self._register_handlers()
        self._populate_installed()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        pet_layout = QHBoxLayout()
        self.pet_label = QLabel("当前桌宠:")
        self.pet_label.setStyleSheet("color: #cdd6f4; font-size: 13px;")
        self.pet_combo = QComboBox()
        self.pet_combo.setFixedHeight(32)
        self.pet_combo.setMinimumWidth(200)
        self.pet_combo.setStyleSheet("""
            QComboBox {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 6px; padding: 0 8px;
            }
            QComboBox::drop-down { border: none; }
        """)

        self.switch_btn = QPushButton("切换")
        self.switch_btn.setFixedHeight(32)
        self.switch_btn.setCursor(Qt.PointingHandCursor)
        self.switch_btn.clicked.connect(self._switch_pet)
        self.switch_btn.setStyleSheet("""
            QPushButton {
                background: #89b4fa; color: #1e1e2e; border: none;
                border-radius: 6px; padding: 0 12px;
            }
            QPushButton:hover { background: #74c7ec; }
        """)

        self.netease_btn = QPushButton("关联网易云音乐")
        self.netease_btn.setFixedHeight(32)
        self.netease_btn.setCursor(Qt.PointingHandCursor)
        self.netease_btn.clicked.connect(self._link_netease)
        self.netease_btn.setStyleSheet("""
            QPushButton {
                background: #ec4141; color: #ffffff; border: none;
                border-radius: 6px; padding: 0 12px;
            }
            QPushButton:hover { background: #d43535; }
        """)

        pet_layout.addWidget(self.pet_label)
        pet_layout.addWidget(self.pet_combo)
        pet_layout.addWidget(self.switch_btn)
        pet_layout.addWidget(self.netease_btn)
        pet_layout.addStretch()
        layout.addLayout(pet_layout)

        self.conn_label = QLabel("未连接")
        self.conn_label.setStyleSheet("color: #f38ba8; font-size: 11px; padding: 4px 0;")
        layout.addWidget(self.conn_label)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setStyleSheet("""
            QTextEdit {
                background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 8px; padding: 8px; font-size: 13px;
            }
        """)
        layout.addWidget(self.chat_log)

        input_layout = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("输入消息...")
        self.input.setFixedHeight(36)
        self.input.setStyleSheet("""
            QLineEdit {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 8px; padding: 0 12px; font-size: 13px;
            }
            QLineEdit:focus { border-color: #89b4fa; }
        """)
        self.input.returnPressed.connect(self._send_msg)

        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedHeight(36)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_msg)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: #a6e3a1; color: #1e1e2e; border: none;
                border-radius: 8px; padding: 0 16px; font-weight: bold;
            }
            QPushButton:hover { background: #94e2d5; }
        """)

        input_layout.addWidget(self.input)
        input_layout.addWidget(self.send_btn)
        layout.addLayout(input_layout)

    def _register_handlers(self):
        @self.socket.on("server_msg")
        def on_server_msg(data):
            if "verification_result" in data:
                success = data.get("success", False)
                self.netease_verified.emit(success)
                color = "#a6e3a1" if success else "#f38ba8"
                self.chat_log.append(f'<span style="color: {color};">{data["msg"]}</span>')
            elif "code_sent" in data:
                self.chat_log.append(f'<span style="color: #89b4fa;">{data["msg"]}</span>')
            else:
                self.chat_log.append(f'<span style="color: #a6e3a1;">桌宠: </span>{data["msg"]}')

            if "cookie_return" in data:
                self.save_cookie(data.get("cookie_return"))
                self.chat_log.append('<span style="color: #89b4fa;">已经更新本地cookie</span>')

    def save_cookie(self, cookie):
        path = "./cookie.json"
        data = {"cookie": cookie}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _populate_installed(self):
        installed = self.manager.list_installed()
        active_id = self.manager.get_active_pet_id()
        self.pet_combo.clear()
        if not installed:
            self.pet_combo.addItem("暂无已安装桌宠")
            self.pet_combo.setEnabled(False)
            return
        self.pet_combo.setEnabled(True)
        for pet_id, info in installed.items():
            self.pet_combo.addItem(f"{pet_id} (v{info['version']})", pet_id)
        if active_id:
            idx = self.pet_combo.findData(active_id)
            if idx >= 0:
                self.pet_combo.setCurrentIndex(idx)

    def _switch_pet(self):
        pet_id = self.pet_combo.currentData()
        if not pet_id:
            return
        self.manager.set_active_pet(pet_id)
        self.socket.select_pet(pet_id)
        self.chat_log.append(f'<span style="color: #89b4fa;">已切换至 {pet_id}</span>')

    def _link_netease(self):
        pet_id = self.pet_combo.currentData() or ""

        cellphone_num = self.manager.get_netease_cellphone()
        if not cellphone_num:
            cellphone_num, ok = QInputDialog.getText(
                self,
                "关联网易云音乐",
                "请输入手机号（首次绑定，后续会自动登录）:",
                QLineEdit.Normal,
            )
            cellphone_num = cellphone_num.strip()
            if not ok or not cellphone_num:
                return
            self.manager.set_netease_cellphone(cellphone_num)

        try:
            result_data = self.socket.check_netease_cookie(cellphone_num)
        except Exception as e:
            self.chat_log.append(f'<span style="color: #f38ba8;">cookie 检查失败: {e}</span>')
            result_data = {"cookie_valid": False}

        if result_data is None:
            self.chat_log.append('<span style="color: #f9e2af;">cookie 检查没有返回结果，请确认服务端已重启到最新代码</span>')
            result_data = {"cookie_valid": False}

        result = bool(result_data.get("cookie_valid", False))
        msg = result_data.get("msg")
        if msg:
            color = "#a6e3a1" if result else "#f9e2af"
            self.chat_log.append(f'<span style="color: {color};">{msg}</span>')
        if "cookie_return" in result_data:
            self.save_cookie(result_data.get("cookie_return"))
            self.chat_log.append('<span style="color: #89b4fa;">已经更新本地cookie</span>')

        if result:
            self.chat_log.append('<span style="color: #a6e3a1;">已自动登录网易云</span>')
            return

        dialog = NeteaseDialog(pet_id, self)
        dialog.id_input.setText(cellphone_num)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.chat_log.append('<span style="color: #89b4fa;">正在提交验证码...</span>')
        self.socket.submit_netease_code(pet_id, cellphone_num, dialog.verified_code)

        v_loop = QEventLoop()
        v_result = None

        def on_verified(success):
            nonlocal v_result
            v_result = success
            v_loop.quit()

        self.netease_verified.connect(on_verified)
        QTimer.singleShot(10000, lambda: v_loop.quit() if v_result is None else None)
        v_loop.exec()
        self.netease_verified.disconnect(on_verified)

        if v_result is None:
            self.chat_log.append('<span style="color: #f9e2af;">验证码超时，请稍后重试</span>')

    def _send_msg(self):
        msg = self.input.text().strip()
        if not msg:
            return
        self.chat_log.append(f'<span style="color: #b4befe;">我: </span>{msg}')
        self.socket.send_chat(msg)
        self.input.clear()

    def connect_socket(self):
        if self.socket.connect():
            self.conn_label.setText("已连接")
            self.conn_label.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 4px 0;")
            self.chat_log.append('<span style="color: #a6e3a1;">已连接服务端</span>')
        else:
            self.conn_label.setText("连接失败 - 聊天功能不可用")
            self.conn_label.setStyleSheet("color: #f38ba8; font-size: 11px; padding: 4px 0;")
            self.chat_log.append('<span style="color: #f38ba8;">无法连接服务端，聊天功能不可用</span>')
