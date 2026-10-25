import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server import socket_handlers, tts_config


class FakeSocketIO:
    def __init__(self):
        self.handlers = {}

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler
        return register


class FakeResponse:
    status_code = 200
    content = b"tts-bytes"

    def raise_for_status(self):
        return None


class TtsPromptTests(unittest.TestCase):
    def setUp(self):
        # These regression tests cover new_api, regardless of local defaults.
        patcher = patch.object(tts_config, 'TTS_BACKEND', 'new_api')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prompt_is_the_complete_text_after_first_underscore(self):
        cases = {
            "normal_どうしてもです。夏木さん.wav": "どうしてもです。夏木さん",
            "happy_一句.带点_额外下划线.wav": "一句.带点_额外下划线",
            "voice_without_prompt.wav": "without_prompt",
            "无扩展名_中文提示": "中文提示",
            "no_underscore.wav": "underscore",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    socket_handlers._prompt_text_from_ref_audio_name(filename),
                    expected,
                )

    def test_voice_produce_sends_filename_prompt_to_tts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "atri" / "config" / "voice_refs"
            reference.mkdir(parents=True)
            filename = "normal_参考文本_保留下划线.wav"
            (reference / filename).write_bytes(b"reference")

            socket = FakeSocketIO()
            socket_handlers.register_socket_handlers(socket)
            response = FakeResponse()
            payload = {
                "data": {
                    "pet_id": "atri",
                    "ref_audio_name": filename,
                    "prompt_text": "",
                    "prompt_lang": "ja",
                    "text": "こんにちは",
                },
            }
            with patch.object(socket_handlers, "SERVER_PETS_DIR", root), patch.object(
                socket_handlers, "_authenticated_username", return_value=("test-user", "")
            ), patch.object(socket_handlers, "request", SimpleNamespace(sid="tts-sid")), patch.object(
                socket_handlers.requests, "post", return_value=response
            ) as post:
                result = socket.handlers["voice_produce"](payload)

            self.assertEqual(result, response.content)
            post.assert_called_once_with(
                url="http://127.0.0.1:9870/tts",
                json={
                    "prompt_text": "参考文本_保留下划线",
                    "prompt_lang": "ja",
                    "text": "こんにちは",
                    "ref_audio_path": str(reference / filename),
                },
                timeout=60,
            )

    def test_explicit_prompt_is_kept_when_filename_has_no_underscore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "atri" / "config" / "voice_refs"
            reference.mkdir(parents=True)
            filename = "normal.wav"
            (reference / filename).write_bytes(b"reference")

            socket = FakeSocketIO()
            socket_handlers.register_socket_handlers(socket)
            response = FakeResponse()
            payload = {
                "data": {
                    "pet_id": "atri",
                    "ref_audio_name": filename,
                    "prompt_text": "客户端传入的提示",
                    "prompt_lang": "ja",
                },
            }
            with patch.object(socket_handlers, "SERVER_PETS_DIR", root), patch.object(
                socket_handlers, "_authenticated_username", return_value=("test-user", "")
            ), patch.object(socket_handlers, "request", SimpleNamespace(sid="tts-sid")), patch.object(
                socket_handlers.requests, "post", return_value=response
            ) as post:
                socket.handlers["voice_produce"](payload)

            self.assertEqual(post.call_args.kwargs["json"]["prompt_text"], "客户端传入的提示")


if __name__ == "__main__":
    unittest.main()
