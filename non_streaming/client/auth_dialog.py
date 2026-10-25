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
    QCheckBox,
    QMessageBox,
)

from api_client import ApiClient


class AuthDialog(QDialog):
    """登录/注册对话框"""

    def __init__(self, api: ApiClient, parent=None):
        super().__init__(parent)
        self.api = api
        self.username: str | None = None
        self.token: str | None = None
        self.password: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("AI 桌宠平台 — 登录")
        self.setFixedSize(320, 300)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QLineEdit {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 6px; padding: 0 10px; height: 32px;
                font-size: 13px;
            }
            QLineEdit:focus { border-color: #89b4fa; }
            QCheckBox { color: #cdd6f4; font-size: 12px; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border-radius: 3px;
                border: 1px solid #45475a; background: #313244;
            }
            QCheckBox::indicator:checked {
                background: #89b4fa; border-color: #89b4fa;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Title
        title = QLabel("登录 / 注册")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Username
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("用户名")
        layout.addWidget(self.user_input)

        # Password
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("密码")
        self.pass_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.pass_input)

        # Remember password checkbox
        self.remember_check = QCheckBox("记住密码（下次自动登录）")
        self.remember_check.setChecked(True)
        layout.addWidget(self.remember_check)

        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #f38ba8; font-size: 11px;")
        layout.addWidget(self.error_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.login_btn = QPushButton("登录")
        self.login_btn.setFixedHeight(36)
        self.login_btn.setCursor(Qt.PointingHandCursor)
        self.login_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.login_btn.setStyleSheet("""
            QPushButton {
                background: #89b4fa; color: #1e1e2e; border: none;
                border-radius: 8px;
            }
            QPushButton:hover { background: #74c7ec; }
        """)
        self.login_btn.clicked.connect(self._on_login)

        self.register_btn = QPushButton("注册")
        self.register_btn.setFixedHeight(36)
        self.register_btn.setCursor(Qt.PointingHandCursor)
        self.register_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.register_btn.setStyleSheet("""
            QPushButton {
                background: #a6e3a1; color: #1e1e2e; border: none;
                border-radius: 8px;
            }
            QPushButton:hover { background: #94e2d5; }
        """)
        self.register_btn.clicked.connect(self._on_register)

        btn_layout.addWidget(self.login_btn)
        btn_layout.addWidget(self.register_btn)
        layout.addLayout(btn_layout)

        # Enter key triggers login
        self.pass_input.returnPressed.connect(self._on_login)

    def set_credentials(self, username: str, password: str = ""):
        self.user_input.setText(username)
        if password:
            self.pass_input.setText(password)

    def _show_error(self, msg: str):
        self.error_label.setText(msg)

    def _on_login(self):
        username = self.user_input.text().strip()
        password = self.pass_input.text()
        if not username or not password:
            self._show_error("请输入用户名和密码")
            return

        try:
            result = self.api.login(username, password)
            self.username = result["username"]
            self.token = result["token"]
            self.password = password
            self.accept()
        except Exception as e:
            self._show_error(str(e))

    def _on_register(self):
        username = self.user_input.text().strip()
        password = self.pass_input.text()
        if not username or not password:
            self._show_error("请输入用户名和密码")
            return

        try:
            result = self.api.register(username, password)
            self.username = result["username"]
            self.token = result["token"]
            self.password = password
            self.accept()
        except Exception as e:
            self._show_error(str(e))
