import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_window import MainWindow


class ScreenRegistrationTests(unittest.TestCase):
    def test_reconnect_registers_pc_screen_source(self):
        socket = Mock()
        socket.call.return_value = {"ok": True}
        window = SimpleNamespace(
            api=SimpleNamespace(token="test-token"),
            harness=SimpleNamespace(available=True),
            socket=socket,
        )
        MainWindow._register_screen_source(window)
        event, payload = socket.call.call_args.args
        self.assertEqual(event, "register_screen_source")
        self.assertEqual(payload, {
            "client_type": "pc",
            "screen_capture": True,
            "file_access": True,
            "deepseek_harness": True,
            "token": "test-token",
        })

    def test_reconnect_callback_only_runs_after_startup(self):
        window = SimpleNamespace(
            _screen_source_startup_complete=False,
            _register_screen_source=Mock(),
        )
        with patch("main_window.threading.Thread") as thread:
            MainWindow._register_screen_source_on_reconnect(window)
            thread.assert_not_called()

            window._screen_source_startup_complete = True
            MainWindow._register_screen_source_on_reconnect(window)
            thread.assert_called_once_with(target=window._register_screen_source, daemon=True)
            thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
