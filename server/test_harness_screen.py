import json
import unittest
from unittest.mock import Mock, patch

from client import bocha_mcp_server
from server import chat_ai, screen_skill, socket_handlers


class FakeSocketIO:
    def __init__(self):
        self.handlers = {}
        self.call = Mock(return_value={"image_base64": "test-image"})

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler
        return register


class HarnessScreenTests(unittest.TestCase):
    def test_mcp_exposes_and_calls_screen_tool(self):
        bridge = Mock()
        bridge.inspect_screen.return_value = '{"activity":"coding"}'
        listing = bocha_mcp_server._handle_request(
            {"id": 1, "method": "tools/list"}, bridge
        )
        names = [tool["name"] for tool in listing["result"]["tools"]]
        self.assertIn("inspect_screen_activity", names)
        screen_tool = next(tool for tool in listing["result"]["tools"] if tool["name"] == "inspect_screen_activity")
        self.assertIn("focus", screen_tool["inputSchema"]["properties"])
        self.assertIn("question", screen_tool["inputSchema"]["properties"])

        result = bocha_mcp_server._handle_request(
            {
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "inspect_screen_activity",
                    "arguments": {
                        "reason": "check editor",
                        "focus": "error dialog",
                        "question": "What does the error say?",
                    },
                },
            },
            bridge,
        )
        bridge.inspect_screen.assert_called_once_with({
            "reason": "check editor",
            "focus": "error dialog",
            "question": "What does the error say?",
        })
        self.assertEqual(result["result"]["content"][0]["text"], '{"activity":"coding"}')

    def test_mcp_forwards_authenticated_pc_request(self):
        bridge = bocha_mcp_server.BochaBridge()
        bridge.socket = Mock(connected=True)
        bridge.socket.call.return_value = {"ok": True, "content": '{"activity":"coding"}'}
        with patch.object(bocha_mcp_server, "SERVER_URL", "https://example.invalid"), patch.object(
            bocha_mcp_server, "USERNAME", "desktop-user"
        ), patch.object(bocha_mcp_server, "AUTH_TOKEN", "test-token"):
            content = bridge.inspect_screen({
                "reason": "check editor",
                "focus": "error dialog",
                "question": "What does the error say?",
            })
        self.assertEqual(content, '{"activity":"coding"}')
        event, payload = bridge.socket.call.call_args.args
        self.assertEqual(event, "harness_screen_inspect")
        self.assertEqual(payload, {
            "reason": "check editor",
            "focus": "error dialog",
            "question": "What does the error say?",
            "username": "desktop-user",
            "client_type": "pc",
            "token": "test-token",
        })

    def test_server_requires_registered_pc_and_captures_screen(self):
        socketio = FakeSocketIO()
        socket_handlers.register_socket_handlers(socketio)
        handler = socketio.handlers["harness_screen_inspect"]
        payload = {
            "username": "desktop-user", "client_type": "pc", "token": "test-token",
            "focus": "error dialog", "question": "What does the error say?",
        }

        def inspect(**kwargs):
            self.assertTrue(kwargs["force_vision"])
            self.assertEqual(kwargs["focus"], "error dialog")
            self.assertEqual(kwargs["question"], "What does the error say?")
            self.assertEqual(
                screen_skill._request_client_screen_capture({"max_width": 720}),
                {"image_base64": "test-image"},
            )
            return '{"activity":"coding"}'

        with patch.object(socket_handlers, "get_authenticated_username", return_value=("desktop-user", "")), patch.dict(
            socket_handlers._SCREEN_SOURCE_SIDS, {}, clear=True
        ), patch.dict(socket_handlers._CLIENT_CAPABILITIES, {}, clear=True), patch.object(
            socket_handlers.chat_ai, "_inspect_screen_activity", side_effect=inspect
        ) as model:
            self.assertFalse(handler(payload)["ok"])
            self.assertFalse(handler({**payload, "client_type": "mobile"})["ok"])
            model.assert_not_called()

            socket_handlers._SCREEN_SOURCE_SIDS["desktop-user"] = "pc-sid"
            socket_handlers._CLIENT_CAPABILITIES["desktop-user"] = {
                "client_type": "pc", "screen_capture": True,
            }
            self.assertEqual(handler(payload), {"ok": True, "content": '{"activity":"coding"}'})
            model.assert_called_once()
            socketio.call.assert_called_once_with(
                "capture_screen", {"max_width": 720}, to="pc-sid", timeout=30
            )
            self.assertIsNone(screen_skill._screen_capture_request_var.get())

    def test_focus_request_reaches_vision_model_and_result(self):
        window = screen_skill.WindowInfo(title="Editor", process_name="editor", process_path="")
        vision_reply = json.dumps({
            "activity": "coding", "intent": "debugging", "confidence": 0.9,
            "should_interrupt": False, "sensitive_visible": False,
            "summary": "Editing code", "focus_found": True,
            "focus_result": "Build failed: missing module",
        })

        def analyze(**kwargs):
            self.assertIn("error dialog", kwargs["prompt"])
            self.assertIn("What does the error say?", kwargs["prompt"])
            return vision_reply

        with patch.object(screen_skill, "idle_seconds", return_value=100000), patch.object(
            screen_skill, "foreground_window_info", return_value=window
        ), patch.object(
            screen_skill, "capture_screen_for_vision", return_value=(b"jpeg", {}, 0.0)
        ), patch.object(
            screen_skill, "analyze_with_openai", side_effect=analyze
        ), patch.object(screen_skill, "mark_vision_model_start"), patch.object(
            screen_skill, "mark_vision_model_finish"
        ), patch.object(screen_skill, "finalize_freshness"):
            result = json.loads(screen_skill.inspect_screen_activity(
                focus="error dialog", question="What does the error say?",
                provider="openai", openai_api_key="test-key",
            ))

        self.assertEqual(result["focus_result"], "Build failed: missing module")
        self.assertTrue(result["focus_found"])
        self.assertEqual(result["provider"], "openai")

    def test_focus_result_is_empty_when_vision_fails(self):
        window = screen_skill.WindowInfo(title="Editor", process_name="editor", process_path="")
        with patch.object(screen_skill, "idle_seconds", return_value=0), patch.object(
            screen_skill, "foreground_window_info", return_value=window
        ), patch.object(
            screen_skill, "capture_screen_for_vision", return_value=(b"jpeg", {}, 0.0)
        ), patch.object(
            screen_skill, "analyze_with_openai", side_effect=RuntimeError("vision failed")
        ), patch.object(screen_skill, "mark_vision_model_start"), patch.object(
            screen_skill, "finalize_freshness"
        ):
            result = json.loads(screen_skill.inspect_screen_activity(
                focus="error dialog", provider="openai", openai_api_key="test-key",
            ))

        self.assertFalse(result["focus_found"])
        self.assertEqual(result["focus_result"], "")
        self.assertIn("vision_error", result)

    def test_gemini_focus_request_uses_same_result_fields(self):
        window = screen_skill.WindowInfo(title="Browser", process_name="browser", process_path="")
        vision_reply = json.dumps({
            "activity": "browsing", "intent": "checking a page", "confidence": 0.8,
            "should_interrupt": False, "sensitive_visible": False,
            "summary": "Viewing a page", "focus_found": True,
            "focus_result": "Upload complete",
        })

        def analyze(**kwargs):
            self.assertIn("status message", kwargs["prompt"])
            self.assertIn("What does it say?", kwargs["prompt"])
            return screen_skill.VisionModelResult(text=vision_reply)

        with patch.object(screen_skill, "idle_seconds", return_value=0), patch.object(
            screen_skill, "foreground_window_info", return_value=window
        ), patch.object(
            screen_skill, "capture_screen_for_vision", return_value=(b"jpeg", {}, 0.0)
        ), patch.object(
            screen_skill, "analyze_with_gemini_key_pool", side_effect=analyze
        ), patch.object(screen_skill, "mark_vision_model_start"), patch.object(
            screen_skill, "mark_vision_model_finish"
        ), patch.object(screen_skill, "finalize_freshness"):
            result = json.loads(screen_skill.inspect_screen_activity(
                focus="status message", question="What does it say?",
                provider="gemini", gemini_api_key="test-key",
            ))

        self.assertTrue(result["focus_found"])
        self.assertEqual(result["focus_result"], "Upload complete")
        self.assertIn("focus_result", screen_skill.GEMINI_SCREEN_RESPONSE_SCHEMA["required"])

    def test_chat_tool_forwards_focus_as_fresh_vision_request(self):
        with patch.object(chat_ai, "_inspect_screen_activity_impl", return_value="{}") as vision:
            chat_ai._inspect_screen_activity(focus="error dialog", question="What does it say?")
        self.assertEqual(vision.call_args.kwargs["focus"], "error dialog")
        self.assertEqual(vision.call_args.kwargs["question"], "What does it say?")
        self.assertTrue(vision.call_args.kwargs["force_vision"])


if __name__ == "__main__":
    unittest.main()
