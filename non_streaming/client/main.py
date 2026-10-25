import sys
import os
from pathlib import Path

# Ensure the client folder itself is on sys.path so sibling imports work
CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from PySide6.QtGui import QFont
import json
from api_client import ApiClient
from auth_dialog import AuthDialog
from main_window import MainWindow
from socket_client import SocketClient
from credentials import (
    save_credentials,
    load_credentials,
    clear_credentials,
    should_auto_login,
)


def get_runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def try_saved_login(api: ApiClient, creds: dict | None) -> bool:
    if not creds:
        return False

    username = (creds.get("username") or "").strip()
    token = creds.get("token") or ""
    password = creds.get("password") or ""
    if not username:
        clear_credentials()
        return False

    if token and api.verify_token(username, token):
        api.token = token
        api.username = username
        return True

    if password:
        try:
            result = api.login(username, password)
        except Exception:
            clear_credentials()
            return False
        save_credentials(result["username"], result["token"], password)
        return True

    clear_credentials()
    return False


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyle("Fusion")

    # Check server reachability
    config_path = get_runtime_dir() / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        server_url = json.load(f)["web_url"]

    api = ApiClient(server_url)
    if not api.is_reachable():
        QMessageBox.critical(
            None, "错误",
            f"无法连接服务端 {server_url}\n请先启动服务器：python server/run.py",
        )
        sys.exit(1)

    # Try auto-login with saved credentials
    saved_creds = load_credentials()
    auto_logged_in = should_auto_login() and try_saved_login(api, saved_creds)

    if not auto_logged_in:
        # Show login/register dialog
        auth = AuthDialog(api)

        # Pre-fill username if saved
        if saved_creds and "username" in saved_creds:
            auth.set_credentials(saved_creds["username"], saved_creds.get("password", ""))

        if auth.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)

        username = auth.username
        token = auth.token
        password = auth.password or ""
        api.username = username
        api.token = token

        # Save credentials if checkbox is checked
        if auth.remember_check.isChecked():
            save_credentials(username, token, password)
        else:
            clear_credentials()

    # Create socket client with username
    socket = SocketClient(server_url, username=api.username or "")

    window = MainWindow(api.username or "", api, socket)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
