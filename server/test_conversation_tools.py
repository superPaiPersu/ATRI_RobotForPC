import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from flask import Flask
from flask_socketio import SocketIO
from werkzeug.serving import make_server

from client.pet_template.AI_tools import SocketAI
from server import chat_ai, conversation_skill, socket_handlers
from server.test_dialogue_history_tools import QuietHandler, make_client, stream_chunk


def action_request(action="save", request_id="test-action", **kwargs):
    return {"request_id": request_id, "username": "test-user", "pet_id": "test-pet", "action": action, **kwargs}


class ConversationActionTests(unittest.TestCase):
    def test_save_and_exit_map_to_false_and_true_and_duplicate_is_ignored(self):
        callback = Mock(return_value=None)
        client = make_client(conversation_action_callback=callback)
        self.assertIn("conversation_action_request", client.sio.handlers["/"])
        request = action_request()
        first = client._on_conversation_action_request(request)
        self.assertTrue(first["ok"])
        self.assertEqual(first["status"], "scheduled")
        self.assertEqual(client._on_conversation_action_request(request), first)
        self.assertFalse(client._on_conversation_action_request(action_request("conversation-over"))["ok"])
        result = client._on_conversation_action_request(action_request("conversation-over", "exit-action"))
        self.assertTrue(result["ok"])
        self.assertEqual(callback.call_args_list, [call(False), call(True)])

    def test_invalid_identity_and_actions_do_not_run_callback(self):
        callback = Mock()
        client = make_client(conversation_action_callback=callback)
        for changes in ({"username": "other"}, {"pet_id": "other"}, {"action": "arbitrary_method"}, {"request_id": ""}):
            self.assertFalse(client._on_conversation_action_request(action_request(**changes))["ok"])
        callback.assert_not_called()

    def test_missing_rejected_and_failed_callbacks_return_errors(self):
        client = make_client()
        self.assertFalse(client._on_conversation_action_request(action_request())["ok"])
        client.set_conversation_action_callback(lambda is_exit: False)
        self.assertFalse(client._on_conversation_action_request(action_request())["ok"])
        callback = Mock(side_effect=RuntimeError("private detail"))
        client.set_conversation_action_callback(callback)
        result = client._on_conversation_action_request(action_request())
        self.assertFalse(result["ok"])
        self.assertNotIn("private detail", str(result))

    def test_capability_is_sent_in_chat_and_realtime_only_after_registration(self):
        client = make_client()
        client.sio = Mock(connected=True, namespaces={"/": "test-sid"})
        client.get_answer([])
        self.assertFalse(client.sio.emit.call_args.args[1]["conversation_actions"])
        client.set_conversation_action_callback(Mock())
        client._stream_active = False
        client.get_answer([])
        self.assertTrue(client.sio.emit.call_args.args[1]["conversation_actions"])
        client.sio.call.return_value = {"ok": True}
        client.start_realtime([])
        self.assertTrue(client.sio.call.call_args.args[1]["conversation_actions"])

    def test_server_actions_report_submission_not_completed_save(self):
        callback = Mock(return_value=None)
        client = make_client(conversation_action_callback=callback)
        socket = Mock()
        socket.call.side_effect = lambda event, data, **kwargs: client._on_conversation_action_request(data)
        requester = socket_handlers._client_conversation_requester(socket, "pet-sid", "test-user", "test-pet")
        token = conversation_skill.set_conversation_action_request(requester)
        try:
            save = json.loads(chat_ai._run_local_tool_by_name("save_conversation", {}))
            exit_result = json.loads(chat_ai._run_gemini_function_call({"name": "end_conversation", "args": {}}))
            self.assertEqual(save, {"ok": True, "status": "scheduled", "action": "save", "is_exit": False})
            self.assertTrue(exit_result["is_exit"])
            self.assertEqual(callback.call_args_list, [call(False), call(True)])
            self.assertEqual(socket.call.call_args.kwargs, {"to": "pet-sid", "timeout": 10})
            socket.call.side_effect = TimeoutError()
            self.assertIn("error", json.loads(conversation_skill.save_conversation()))
            socket.call.side_effect = None
            socket.call.return_value = {"request_id": "wrong", "ok": True}
            self.assertFalse(json.loads(conversation_skill.save_conversation())["ok"])
        finally:
            conversation_skill.reset_conversation_action_request(token)
        self.assertFalse(json.loads(conversation_skill.end_conversation())["ok"])

    def test_ordinary_chat_disables_actions_for_unregistered_clients(self):
        socket = Mock()
        socket.handlers = {}
        socket.on = lambda event: lambda handler: socket.handlers.setdefault(event, handler)
        socket_handlers.register_socket_handlers(socket)

        def reply(*args, **kwargs):
            names = {tool["name"] for tool in chat_ai._active_local_tool_definitions()}
            self.assertFalse(socket_handlers.CONVERSATION_ACTION_TOOL_NAMES & names)
            return {"content": "normal reply"}

        with patch.object(socket_handlers, "request", SimpleNamespace(sid="mobile-sid")), patch.object(
            socket_handlers, "_authenticated_username", return_value=("test-user", "")
        ), patch.object(chat_ai, "get_answer", side_effect=reply):
            self.assertEqual(socket.handlers["get_answer"]({
                "username": "test-user", "pet_id": "test-pet", "model_name": "gpt", "data": [],
            }), {"content": "normal reply"})
        socket.call.assert_not_called()
        self.assertIsNone(conversation_skill._conversation_action_request_var.get())

    def test_realtime_worker_uses_same_two_actions_for_qwen_and_gemini(self):
        for provider, start_name in (("qwen", "qwen_realtime_start"), ("gemini-live", "gemini_realtime_start")):
            with self.subTest(provider=provider):
                socket = Mock()
                socket.handlers = {}
                socket.on = lambda event: lambda handler: socket.handlers.setdefault(event, handler)
                socket_handlers.register_socket_handlers(socket)
                callback = Mock(return_value=None)
                client = make_client(conversation_action_callback=callback)
                socket.call.side_effect = lambda event, data, **kwargs: client._on_conversation_action_request(data)
                session = SimpleNamespace(model="test-realtime", transcription_model="test-stt", close=Mock())
                with patch.object(socket_handlers, "request", SimpleNamespace(sid="voice-sid")), patch.object(
                    socket_handlers, "_authenticated_username", return_value=("test-user", "")
                ), patch.dict(socket_handlers._REALTIME_SESSIONS, {}, clear=True), patch.dict(
                    socket_handlers._REALTIME_AUDIO_COUNTS, {}, clear=True
                ), patch.object(chat_ai, start_name, return_value=session) as start:
                    response = socket.handlers["start_realtime"]({
                        "username": "test-user", "pet_id": "test-pet", "realtime_provider": provider,
                        "conversation_actions": True, "data": [],
                    })
                    self.assertTrue(response["ok"], response)
                    self.assertTrue(socket_handlers.CONVERSATION_ACTION_TOOL_NAMES <= set(response["tools"]))
                    runner = start.call_args.kwargs["tool_runner"]
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        result = json.loads(pool.submit(runner, "end_conversation", {}).result(timeout=5))
                        self.assertTrue(result["is_exit"])
                        self.assertIsNone(pool.submit(conversation_skill._conversation_action_request_var.get).result())
                    callback.assert_called_once_with(True)


class ConversationSocketIntegrationTests(unittest.TestCase):
    def test_real_socket_delivers_both_actions_during_model_tool_loop(self):
        app = Flask(__name__)
        app.config["SECRET_KEY"] = "test-only"
        socket = SocketIO(app, async_mode="threading", ping_interval=0.2, ping_timeout=2)
        socket_handlers.register_socket_handlers(socket)
        http_server = make_server("127.0.0.1", 0, app, threaded=True, request_handler=QuietHandler)
        worker = threading.Thread(target=http_server.serve_forever, daemon=True)
        worker.start()
        callback = Mock(return_value=None)
        model = Mock()
        model.chat.completions.create.side_effect = [
            iter([stream_chunk(name="save_conversation")]),
            iter([stream_chunk(text='{"action":"reply","cn":"已提交保存。","jp":"保存を依頼したよ。"}')]),
            iter([stream_chunk(name="end_conversation")]),
            iter([stream_chunk(text='{"action":"reply","cn":"已提交退出。","jp":"終了を依頼したよ。"}')]),
        ]
        completed = threading.Event()

        def receive(chunk):
            if chunk[0] == "end":
                completed.set()

        def connect(client):
            client.sio.connect(client.base_url, transports=["polling"], wait_timeout=5)

        client = None
        try:
            with patch.object(SocketAI, "_connect", connect), patch.object(
                chat_ai, "_create_openai_client", return_value=model
            ), patch.object(socket_handlers, "REQUIRE_SOCKET_AUTH", False):
                client = SocketAI(
                    base_url=f"http://127.0.0.1:{http_server.server_port}", username="test-user", pet_id="test-pet",
                    chunk_sink=receive, conversation_action_callback=callback,
                )
                client.model_name, client.api_url, client.api_key = "gpt", "https://example.invalid/v1", "test-key"
                for text, expected in (("保存", False), ("退出", True)):
                    completed.clear()
                    client.get_answer([{"role": "user", "content": text}])
                    self.assertTrue(completed.wait(timeout=8))
                    # The sink sees end before SocketAI finishes resetting the
                    # stream flag. Wait for that handler to return before the
                    # test starts the next turn immediately.
                    deadline = time.monotonic() + 1
                    while client._stream_active and time.monotonic() < deadline:
                        time.sleep(0.001)
                    self.assertFalse(client._stream_active)
                    callback.assert_called_with(expected)
                    sent = model.chat.completions.create.call_args.kwargs["messages"]
                    tool_result = json.loads(next(message["content"] for message in sent if message["role"] == "tool"))
                    self.assertEqual(tool_result["status"], "scheduled")
                self.assertEqual(callback.call_args_list, [call(False), call(True)])
        finally:
            if client is not None:
                client.sio.disconnect()
            http_server.shutdown()
            http_server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
