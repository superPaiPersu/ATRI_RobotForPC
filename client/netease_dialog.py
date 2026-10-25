from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)


class NeteaseDialog(QDialog):
    """网易云音乐绑定对话框"""

    def __init__(self, pet_id: str, parent=None):
        super().__init__(parent)
        self.pet_id = pet_id
        self.verified_code: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        title = f"关联网易云音乐 — {self.pet_id}" if self.pet_id else "关联网易云音乐"
        self.setWindowTitle(title)
        self.setFixedSize(360, 220)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; font-size: 12px; }
            QLineEdit {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 6px; padding: 0 10px; height: 32px;
                font-size: 13px;
            }
            QLineEdit:focus { border-color: #89b4fa; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # NetEase ID row
        id_layout = QHBoxLayout()
        id_label = QLabel("网易云ID:")
        id_label.setFixedWidth(70)
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("请输入你的网易云ID")
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.id_input)
        layout.addLayout(id_layout)

        # Verification code row
        code_layout = QHBoxLayout()
        code_label = QLabel("验证码:")
        code_label.setFixedWidth(70)
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("点击右侧按钮获取验证码后填写")
        self.verify_btn = QPushButton("获取验证码")
        self.verify_btn.setFixedHeight(32)
        self.verify_btn.setCursor(Qt.PointingHandCursor)
        self.verify_btn.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background: #f9e2af; color: #1e1e2e; border: none;
                border-radius: 6px; padding: 0 10px;
            }
            QPushButton:hover { background: #f2cc8f; }
        """)
        self.verify_btn.clicked.connect(self._on_get_code)
        code_layout.addWidget(code_label)
        code_layout.addWidget(self.code_input)
        code_layout.addWidget(self.verify_btn)
        layout.addLayout(code_layout)

        # Confirm button
        self.confirm_btn = QPushButton("确认关联")
        self.confirm_btn.setFixedHeight(36)
        self.confirm_btn.setCursor(Qt.PointingHandCursor)
        self.confirm_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background: #ec4141; color: #ffffff; border: none;
                border-radius: 8px;
            }
            QPushButton:hover { background: #d43535; }
        """)
        self.confirm_btn.clicked.connect(self._on_confirm)
        layout.addWidget(self.confirm_btn)

    def _on_get_code(self):
        """获取验证码按钮点击 — 向服务器请求发送验证码"""
        netease_id = self.id_input.text().strip()
        if not netease_id:
            self.id_input.setPlaceholderText("请先输入网易云ID")
            return
        # 通知服务器发送验证码
        from chat_panel import ChatPanel
        parent = self.parent()
        if isinstance(parent, ChatPanel):
            parent.socket.request_netease_code(self.pet_id, netease_id)
        self.code_input.setPlaceholderText("验证码已发送，请查收后填写")

    def _on_confirm(self):
        code = self.code_input.text().strip()
        if not code:
            return
        self.verified_code = code
        self.accept()
