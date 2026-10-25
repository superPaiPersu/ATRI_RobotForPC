import copy
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask
from flask_socketio import SocketIO
from werkzeug.serving import WSGIRequestHandler, make_server

from client.pet_template.AI_tools import SocketAI
from server import chat_ai, history_skill, socket_handlers


TAGS = ["睡眠"]
MEMORY = [{"role": "user", "content": "我习惯熬夜。"}]


def make_client(callback=None, **kwargs):
    with patch.object(SocketAI, "_connect"):
        return SocketAI(username="test-user", pet_id="test-pet", history_search_callback=callback, **kwargs)


def history_request(**kwargs):
    return {"request_id": "test-request", "username": "test-user", "pet_id": "test-pet", "tags": TAGS, **kwargs}


class ClientHistoryTests(unittest.TestCase):
    def test_template_session_ids_are_resolved_over_socket_listener(self):
        from client.test_history_search import make_template_owner
        for count in (1, 6):
            with self.subTest(count=count):
                owner = make_template_owner([TAGS] + [[]] * (count - 1))
                client = make_client(owner.tags_match)
                result = client._on_dialogue_history_request(history_request())
                self.assertEqual(result, {"request_id": "test-request", "messages": owner.tags_index_dict[0], "matched": True})

    def test_listener_uses_live_callback_and_returns_only_matches(self):
        callback = Mock(return_value=copy.deepcopy(MEMORY))
        client = make_client(callback)
        self.assertIn("dialogue_history_request", client.sio.handlers["/"])
        result = client._on_dialogue_history_request(history_request())
        self.assertEqual(result, {"request_id": "test-request", "messages": MEMORY, "matched": True})
        callback.assert_called_once_with(TAGS)
        callback.return_value = [{"role": "assistant", "content": "后续新增的历史。"}]
        self.assertEqual(client._on_dialogue_history_request(history_request())["messages"], callback.return_value)

    def test_identity_and_invalid_tags_do_not_execute_callback(self):
        callback = Mock()
        client = make_client(callback)
        for change in ({"username": "other"}, {"pet_id": "other"}, {"request_id": ""}, {"tags": []}, {"tags": [["sleep", "habit"]]}):
            with self.subTest(change=change):
                self.assertIn("error", client._on_dialogue_history_request(history_request(**change)))
        callback.assert_not_called()

    def test_empty_results_and_callback_errors_remain_distinct(self):
        callback = Mock()
        client = make_client(callback)
        for value in ([], None, [{"role": "developer", "content": "没有匹配历史。"}], {"messages": MEMORY, "matched": False}):
            callback.return_value = value
            result = client._on_dialogue_history_request(history_request())
            self.assertEqual(result["messages"], [])
            self.assertFalse(result["matched"])
        callback.side_effect = KeyError(None)
        self.assertFalse(client._on_dialogue_history_request(history_request())["matched"])
        callback.side_effect = RuntimeError("private data")
        result = client._on_dialogue_history_request(history_request())
        self.assertIn("error", result)
        self.assertNotIn("private data", str(result))
        self.assertNotIn("matched", result)

    def test_oversized_or_malformed_result_is_not_returned(self):
        callback = Mock()
        client = make_client(callback)
        for value in ("not a list", [None], [{"role": "user", "content": "x" * (2 * 1024 * 1024)}]):
            callback.return_value = value
            result = client._on_dialogue_history_request(history_request())
            self.assertIn("error", result)
            self.assertNotIn("messages", result)

    def test_capability_follows_registered_callback(self):
        client = make_client()
        client.sio = Mock(connected=True)
        client.sio.namespaces = {"/": "test-sid"}
        client.get_answer([])
        self.assertFalse(client.sio.emit.call_args.args[1]["dialogue_history"])
        client._stream_active = False
        client.set_history_search_callback(lambda tags: [])
        client.get_answer([])
        self.assertTrue(client.sio.emit.call_args.args[1]["dialogue_history"])
        client.sio.call.return_value = {"ok": True}
        client.start_realtime([])
        self.assertTrue(client.sio.call.call_args.args[1]["dialogue_history"])
        client.set_history_search_callback(None)
        self.assertIn("error", client._on_dialogue_history_request(history_request()))


class ServerHistoryTests(unittest.TestCase):
    def setUp(self):
        self.registry = patch.object(history_skill, "dialogue_tag_lists", return_value=["睡眠"])
        self.registry.start()
        self.addCleanup(self.registry.stop)

    def test_vocabulary_is_read_locally_without_reverse_request(self):
        requester = Mock(side_effect=AssertionError("unexpected network request"))
        token = history_skill.set_history_search_request(requester)
        try:
            vocabulary = json.loads(history_skill.get_dialogue_tags())
            self.assertTrue(all(isinstance(tag, str) for tag in vocabulary))
            self.assertEqual(vocabulary, ["睡眠"])
            requester.assert_not_called()
        finally:
            history_skill.reset_history_search_request(token)

    def test_search_validates_exact_labels_and_returns_only_requested_results(self):
        requester = Mock(return_value={"messages": MEMORY, "matched": True})
        token = history_skill.set_history_search_request(requester)
        try:
            result = json.loads(history_skill.search_dialogue_history(TAGS + TAGS))
            self.assertEqual(result, {"messages": MEMORY, "matched": True})
            requester.assert_called_once_with({"tags": TAGS})
            for tags in ([], ["not-an-existing-concept"], [["睡眠", "习惯"]], [123], ["  "]):
                with self.assertRaises(ValueError):
                    history_skill.search_dialogue_history(tags)
            requester.assert_called_once()
            requester.return_value = {"error": "temporarily unavailable"}
            self.assertEqual(json.loads(history_skill.search_dialogue_history(TAGS)), requester.return_value)
        finally:
            history_skill.reset_history_search_request(token)
        self.assertIn("error", json.loads(history_skill.search_dialogue_history(TAGS)))

    def test_request_ids_timeout_and_target_are_checked(self):
        socket = Mock()
        request = socket_handlers._client_history_requester(socket, "origin-sid", "test-user", "test-pet")

        def reply(event, data, **kwargs):
            self.assertEqual(event, "dialogue_history_request")
            self.assertEqual(kwargs, {"to": "origin-sid", "timeout": 20})
            return {"request_id": data["request_id"], "messages": MEMORY, "matched": True}

        socket.call.side_effect = reply
        self.assertEqual(request({"tags": TAGS})["messages"], MEMORY)
        first_id = socket.call.call_args.args[1]["request_id"]
        request({"tags": TAGS})
        self.assertNotEqual(socket.call.call_args.args[1]["request_id"], first_id)
        socket.call.side_effect = None
        socket.call.return_value = {"request_id": "wrong", "messages": MEMORY}
        self.assertIn("error", request({"tags": TAGS}))
        socket.call.side_effect = TimeoutError()
        result = request({"tags": TAGS})
        self.assertIn("error", result)
        self.assertNotIn("matched", result)

    def test_all_provider_schemas_and_adapters_include_search(self):
        names = {tool["function"]["name"] for tool in chat_ai._active_deepseek_tool_definitions()}
        self.assertTrue(socket_handlers.HISTORY_SEARCH_TOOL_NAMES <= names)
        gemini_names = {tool["name"] for tool in chat_ai._active_gemini_tool_declarations()[0]["functionDeclarations"]}
        self.assertTrue(socket_handlers.HISTORY_SEARCH_TOOL_NAMES <= gemini_names)
        token = history_skill.set_history_search_request(lambda payload: {"messages": MEMORY})
        try:
            chat_result = chat_ai._run_chat_completion_tool_call({"function": {
                "name": "search_dialogue_history", "arguments": json.dumps({"tags": TAGS}),
            }})
            gemini_result = chat_ai._run_gemini_function_call({"name": "search_dialogue_history", "args": {"tags": TAGS}})
            self.assertEqual(json.loads(chat_result)["messages"], MEMORY)
            self.assertEqual(json.loads(gemini_result)["messages"], MEMORY)
        finally:
            history_skill.reset_history_search_request(token)

    def test_realtime_tool_worker_restores_request_context(self):
        socket = Mock()
        socket.handlers = {}
        socket.on = lambda event: lambda handler: socket.handlers.setdefault(event, handler)
        socket_handlers.register_socket_handlers(socket)
        client = make_client(lambda tags: MEMORY)
        socket.call.side_effect = lambda event, data, **kwargs: client._on_dialogue_history_request(data)
        session = SimpleNamespace(model="test-qwen", transcription_model="test-stt", close=Mock())
        with patch.object(socket_handlers, "request", SimpleNamespace(sid="voice-sid")), patch.object(
            socket_handlers, "_authenticated_username", return_value=("test-user", "")
        ), patch.dict(socket_handlers._REALTIME_SESSIONS, {}, clear=True), patch.dict(
            socket_handlers._REALTIME_AUDIO_COUNTS, {}, clear=True
        ), patch.object(chat_ai, "qwen_realtime_start", return_value=session) as start:
            result = socket.handlers["start_realtime"]({
                "username": "test-user", "pet_id": "test-pet", "realtime_provider": "qwen",
                "dialogue_history": True, "data": [],
            })
            self.assertTrue(result["ok"], result)
            self.assertIn("search_dialogue_history", result["tools"])
            runner = start.call_args.kwargs["tool_runner"]
            with ThreadPoolExecutor(max_workers=1) as pool:
                output = pool.submit(runner, "search_dialogue_history", {"tags": TAGS}).result(timeout=5)
                self.assertEqual(json.loads(output)["messages"], MEMORY)
                self.assertIsNone(pool.submit(history_skill._history_search_request_var.get).result())
            self.assertEqual(socket.call.call_args.kwargs["to"], "voice-sid")

    def test_chat_without_callback_keeps_new_tools_disabled(self):
        socket = Mock()
        socket.handlers = {}
        socket.on = lambda event: lambda handler: socket.handlers.setdefault(event, handler)
        socket_handlers.register_socket_handlers(socket)

        def answer(*args, **kwargs):
            names = {tool["name"] for tool in chat_ai._active_local_tool_definitions()}
            self.assertFalse(socket_handlers.HISTORY_SEARCH_TOOL_NAMES & names)
            self.assertIn("get_all_dialogue_history", names)
            return {"content": "normal mobile reply"}

        with patch.object(socket_handlers, "request", SimpleNamespace(sid="mobile-sid")), patch.object(
            socket_handlers, "_authenticated_username", return_value=("test-user", "")
        ), patch.object(chat_ai, "get_answer", side_effect=answer):
            result = socket.handlers["get_answer"]({
                "username": "test-user", "pet_id": "test-pet", "client_type": "mobile",
                "model_name": "gemini", "data": [],
            })
        self.assertEqual(result["content"], "normal mobile reply")
        socket.call.assert_not_called()
        self.assertIsNone(history_skill._history_search_request_var.get())

    def test_pc_agent_branch_uses_originating_pet_socket(self):
        socket = Mock()
        socket.handlers = {}
        socket.on = lambda event: lambda handler: socket.handlers.setdefault(event, handler)
        socket_handlers.register_socket_handlers(socket)
        client = make_client(lambda tags: MEMORY)
        socket.call.side_effect = lambda event, data, **kwargs: client._on_dialogue_history_request(data)

        def task_turn(*args, **kwargs):
            return json.loads(chat_ai._run_local_tool_by_name("search_dialogue_history", {"tags": TAGS}))

        with patch.object(socket_handlers, "request", SimpleNamespace(sid="pet-chat-sid")), patch.object(
            socket_handlers, "_authenticated_username", return_value=("test-user", "")
        ), patch.dict(socket_handlers._SCREEN_SOURCE_SIDS, {"test-user": "main-client-sid"}, clear=True), patch.dict(
            socket_handlers._CLIENT_CAPABILITIES,
            {"test-user": {"client_type": "pc", "file_access": True}}, clear=True,
        ), patch.object(socket_handlers.task_agent, "run_task_turn", side_effect=task_turn):
            result = socket.handlers["get_answer"]({
                "username": "test-user", "pet_id": "test-pet", "client_type": "pc",
                "model_name": "deepseek", "dialogue_history": True, "data": [],
            })
        self.assertEqual(result["messages"], MEMORY)
        self.assertEqual(socket.call.call_args.kwargs["to"], "pet-chat-sid")
        self.assertIsNone(history_skill._history_search_request_var.get())


class QuietHandler(WSGIRequestHandler):
    def log_request(self, *args):
        pass


def stream_chunk(name=None, arguments=None, text=None):
    tool_calls = None if name is None else [SimpleNamespace(
        index=0, id="call-" + name, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {}, ensure_ascii=False)),
    )]
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, tool_calls=tool_calls), finish_reason=None,
    )])


class SocketHistoryIntegrationTests(unittest.TestCase):
    def test_real_socket_acknowledgement_completes_model_tool_loop(self):
        app = Flask(__name__)
        app.config["SECRET_KEY"] = "test-only"
        socket = SocketIO(app, async_mode="threading", ping_interval=0.2, ping_timeout=2)
        socket_handlers.register_socket_handlers(socket)
        http_server = make_server("127.0.0.1", 0, app, threaded=True, request_handler=QuietHandler)
        worker = threading.Thread(target=http_server.serve_forever, daemon=True)
        worker.start()
        completed = threading.Event()
        chunks = []
        callback = Mock(return_value=copy.deepcopy(MEMORY))
        model = Mock()
        model.chat.completions.create.side_effect = [
            iter([stream_chunk(name="get_dialogue_tags")]),
            iter([stream_chunk(name="search_dialogue_history", arguments={"tags": TAGS})]),
            iter([stream_chunk(text='{"action":"reply","cn":"你习惯熬夜。","jp":"夜更かしの習慣があるね。"}')]),
        ]

        def receive(chunk):
            chunks.append(chunk)
            if chunk[0] == "end":
                completed.set()

        def connect(client):
            client.sio.connect(client.base_url, transports=["polling"], wait_timeout=5)

        client = None
        try:
            with patch.object(history_skill, "dialogue_tag_lists", return_value=["睡眠"]), patch.object(SocketAI, "_connect", connect), patch.object(
                chat_ai, "_create_openai_client", return_value=model
            ), patch.object(socket_handlers, "REQUIRE_SOCKET_AUTH", False), patch.dict(
                socket_handlers._SCREEN_SOURCE_SIDS, {"test-user": "other-desktop-sid"}, clear=True
            ), patch.dict(socket_handlers._CLIENT_CAPABILITIES, {}, clear=True):
                client = SocketAI(
                    base_url=f"http://127.0.0.1:{http_server.server_port}", username="test-user", pet_id="test-pet",
                    chunk_sink=receive, history_search_callback=callback,
                )
                client.model_name = "gpt"
                client.api_url = "https://example.invalid/v1"
                client.api_key = "test-key"
                client.get_answer([{"role": "user", "content": "我有什么睡眠习惯？"}])
                self.assertTrue(completed.wait(timeout=8), chunks)
                callback.assert_called_once_with(TAGS)
                self.assertEqual(model.chat.completions.create.call_count, 3)
                sent = model.chat.completions.create.call_args.kwargs["messages"]
                outputs = [message["content"] for message in sent if message.get("role") == "tool"]
                self.assertEqual(len(outputs), 2)
                self.assertEqual(json.loads(outputs[-1]), {"messages": MEMORY, "matched": True})
                self.assertIn("睡眠", json.loads(outputs[0]))
                self.assertFalse(any("request_id" in str(chunk) for chunk in chunks))
                self.assertIsNone(history_skill._history_search_request_var.get())
        finally:
            if client is not None:
                client.sio.disconnect()
            http_server.shutdown()
            http_server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
