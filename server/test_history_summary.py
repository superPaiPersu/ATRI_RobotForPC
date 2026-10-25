import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask
from flask_socketio import SocketIO

from server import chat_ai, socket_handlers


class FakeSocketIO:
    def __init__(self):
        self.handlers = {}
        self.jobs = []

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler
        return register

    def start_background_task(self, function, *args):
        self.jobs.append((function, args))

    def run_next(self):
        function, args = self.jobs.pop(0)
        function(*args)


def session_content(text="I often stay up late", started_at="2026-10-21 12:00:00"):
    events = [
        {"type": "dialogue_session_start", "started_at": started_at, "messages": []},
        {"type": "dialogue_message", "messages": [{"role": "user", "content": text}]},
        {"type": "dialogue_session_end", "ended_at": "2026-10-21 12:10:00", "messages": []},
    ]
    return "".join(json.dumps(event) + "\n" for event in events)


class AutomaticHistorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.socketio = FakeSocketIO()
        socket_handlers.register_socket_handlers(self.socketio)
        self.auth = patch.object(socket_handlers, "_authenticated_username", return_value=("test-user", ""))
        self.path = patch.object(socket_handlers, "DIR_PATH", str(self.root))
        self.tags = patch.object(socket_handlers, "_dialogue_tag_lists", return_value=["sleep"])
        self.auth.start()
        self.path.start()
        self.tags.start()
        socket_handlers._HISTORY_SUMMARY_PENDING.clear()
        self.payload = {
            "username": "test-user", "pet_id": "test-pet",
            "model_name": "deepseek", "api_url": "https://example.invalid/v1",
            "api_key": "test-key", "data": session_content(),
        }
        self.save = self.socketio.handlers["save_dialogue_history"]
        self.model = patch.object(chat_ai, "get_answer")
        self.answer = self.model.start()
        self.answer.return_value = {"content": json.dumps({
            "action": "summary", "user": "User often stays up late.", "model": "",
            "tags": ["sleep", "sleep", "rabbit"],
        })}

    def tearDown(self):
        self.model.stop()
        self.tags.stop()
        self.path.stop()
        self.auth.stop()
        socket_handlers._HISTORY_SUMMARY_PENDING.clear()
        self.temporary.cleanup()

    def events(self, filename):
        path = self.root / "users_data" / "test-user" / "test-pet" / filename
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_tags_are_generated_before_history_is_written(self):
        disabled_before = chat_ai._active_local_tool_definitions()

        def answer(*args):
            history_path = self.root / "users_data" / "test-user" / "test-pet" / "chat_history.txt"
            self.assertFalse(history_path.exists())
            self.assertEqual(chat_ai._active_local_tool_definitions(), [])
            return {"content": json.dumps({
                "action": "summary", "user": "User stays up late.", "model": "",
                "tags": ["sleep", "sleep", "rabbit"],
            })}

        self.answer.side_effect = answer
        self.assertEqual(self.save(self.payload), "success")
        self.answer.assert_called_once()
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], ["sleep", "rabbit"])
        summary_events = self.events("summary_history.txt")
        self.assertEqual(summary_events[-1]["tags"], ["sleep", "rabbit"])
        self.assertEqual(summary_events[-1]["summary_source_id"], self.events("chat_history.txt")[-1]["summary_source_id"])
        self.assertEqual(self.answer.call_args.args[:3], ("deepseek", "https://example.invalid/v1", "test-key"))
        self.assertEqual(chat_ai._active_local_tool_definitions(), disabled_before)

    def test_duplicate_save_and_legacy_summary_do_not_duplicate_work(self):
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.socketio.jobs, [])
        self.assertEqual(len(self.events("chat_history.txt")), 3)
        before = self.events("summary_history.txt")
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.socketio.handlers["update_summary_history"](self.payload), "success")
        self.assertEqual(self.socketio.jobs, [])
        self.assertEqual(self.events("summary_history.txt"), before)
        self.answer.assert_called_once()

    def test_failure_keeps_original_and_same_save_can_retry(self):
        self.answer.side_effect = RuntimeError("upstream unavailable")
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.events("chat_history.txt")[1]["messages"][0]["content"], "I often stay up late")
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], [])
        self.assertEqual(socket_handlers._HISTORY_SUMMARY_PENDING, set())
        self.answer.side_effect = None
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(len(self.events("chat_history.txt")), 3)
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], ["sleep", "rabbit"])


    def test_interleaved_sessions_do_not_overwrite_later_history(self):
        self.save(self.payload)
        other = {**self.payload, "data": session_content("Another session", "2026-10-21 13:00:00")}
        self.save(other)
        self.assertEqual(len(self.events("chat_history.txt")), 6)
        events = self.events("chat_history.txt")
        self.assertEqual(len(events), 6)
        self.assertEqual(events[4]["messages"][0]["content"], "Another session")
        self.assertEqual(events[2]["tags"], ["sleep", "rabbit"])
        self.assertEqual(events[5]["tags"], ["sleep", "rabbit"])

    def test_invalid_jsonl_or_unauthenticated_save_never_queues(self):
        self.assertTrue(self.save({**self.payload, "data": "not-json"}).startswith("failed----"))
        self.assertTrue(self.save({**self.payload, "data": '{"type":"dialogue_message","messages":[]}'}).startswith("failed----"))
        with patch.object(socket_handlers, "_authenticated_username", return_value=(None, "denied")):
            self.assertEqual(self.save(self.payload), "failed----denied")
        self.assertEqual(self.socketio.jobs, [])
        self.answer.assert_not_called()

    def test_no_sense_and_heartbeat_do_not_produce_fake_memories(self):
        self.save({**self.payload, "data": session_content("heart_beat")})
        self.answer.assert_not_called()
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], [])
        self.assertEqual(self.events("summary_history.txt")[-1]["action"], "no_sense")
        self.assertFalse(any(event["type"] == "summary_message" for event in self.events("summary_history.txt")))

    def test_normalizer_retains_tags_for_all_summary_providers(self):
        result = chat_ai._normalize_summary_payload({
            "action": "summary", "user": "User stays up late.", "model": "",
            "tags": ["sleep", "rabbit"],
        })
        self.assertEqual(result["tags"], ["sleep", "rabbit"])

    def test_summary_normalizer_keeps_speakers_separate(self):
        for user, model in (("用户习惯熬夜。", ""), ("", "助手答应提醒休息。")):
            with self.subTest(user=user, model=model):
                raw = json.dumps({"action": "summary", "user": user, "model": model, "tags": ["sleep", "rabbit"]})
                normalized = chat_ai._finalize_model_content(raw, summary_mode=True)
                result = json.loads(normalized)
                self.assertEqual(result["user"], user)
                self.assertEqual(result["model"], model)
        self.answer.return_value = {"content": chat_ai._finalize_model_content(json.dumps({
            "action": "summary", "user": "用户习惯熬夜。", "model": "", "tags": ["sleep", "rabbit"],
        }), summary_mode=True)}
        self.save(self.payload)
        messages = [message for event in self.events("summary_history.txt") for message in event["messages"]]
        self.assertEqual(messages, [{"role": "user", "content": "用户习惯熬夜。"}])

    def test_no_sense_still_indexes_original_dialogue(self):
        raw = json.dumps({"action": "no_sense", "tags": ["sleep", "rabbit"]})
        normalized = chat_ai._finalize_model_content(raw, summary_mode=True)
        self.assertEqual(json.loads(normalized)["tags"], ["sleep", "rabbit"])
        self.assertEqual(chat_ai._deepseek_stream_result_to_payload(raw, True)["tags"], ["sleep", "rabbit"])
        self.answer.return_value = {"content": normalized}
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], ["sleep", "rabbit"])
        summary = self.events("summary_history.txt")
        self.assertEqual(summary[-1]["action"], "no_sense")
        self.assertEqual(summary[-1]["tags"], ["sleep", "rabbit"])
        self.assertFalse(any(event["type"] == "summary_message" for event in summary))

    def test_time_context_and_wait_are_passed_without_heartbeat(self):
        events = [
            {"type": "dialogue_session_start", "started_at": "2026-10-21 12:00:00", "marker": "PC", "messages": []},
            {"type": "dialogue_message", "messages": [{"role": "user", "content": "2026-10-21 12:01:00 | 昨晚没熬夜。", "timestamp": "2026-10-21 12:01:00"}]},
            {"type": "dialogue_message", "messages": [{"role": "model", "content": "记住了。"}]},
            {"type": "dialogue_message", "messages": [{"role": "user", "content": "2026-10-21 12:02:00 | heart_beat"}]},
            {"type": "dialogue_message", "messages": [{"role": "assistant", "content": "wait", "action": "wait"}]},
            {"type": "dialogue_session_end", "ended_at": "2026-10-21 12:10:00", "messages": []},
        ]
        content = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
        self.assertEqual(self.save({**self.payload, "data": content}), "success")
        prompt = self.answer.call_args.args[3]
        data = json.loads(prompt[1]["content"])
        self.assertEqual(data["session"], {"started_at": events[0]["started_at"], "ended_at": events[-1]["ended_at"], "marker": "PC"})
        self.assertEqual(data["dialogue"], [events[1]["messages"][0], {"role": "assistant", "content": "记住了。"}, events[4]["messages"][0]])
        self.assertTrue(chat_ai.is_summary_turn(prompt))
        saved = self.events("chat_history.txt")
        self.assertEqual(saved[3]["messages"], events[3]["messages"])
        self.assertEqual(saved[4]["messages"], events[4]["messages"])

    def test_wait_only_is_saved_and_sent_for_summary_and_tags(self):
        messages = [
            {"role": "user", "content": "wait"},
            {"role": "assistant", "content": "wait"},
            {"role": "assistant", "content": "2026-10-21 12:01:00 | wait"},
            {"role": "model", "content": "wait", "action": "wait"},
            {"role": "assistant", "content": "等你忙完我们再聊熬夜的事。", "action": "wait"},
        ]
        for message in messages:
            with self.subTest(message=message):
                events = [
                    {"type": "dialogue_session_start", "started_at": "2026-10-21 12:00:00", "messages": []},
                    {"type": "dialogue_message", "messages": [message]},
                    {"type": "dialogue_session_end", "ended_at": "2026-10-21 12:10:00", "messages": []},
                ]
                content = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
                self.answer.reset_mock()
                self.assertEqual(self.save({**self.payload, "data": content}), "success")
                self.answer.assert_called_once()
                prompt = self.answer.call_args.args[3]
                expected = {**message, "role": "assistant" if message["role"] == "model" else message["role"]}
                self.assertEqual(json.loads(prompt[1]["content"])["dialogue"], [expected])
                self.assertIn("wait 消息也参与总结和标签提取参考", prompt[0]["content"])
                self.assertEqual(self.events("chat_history.txt")[-2]["messages"], [message])
                self.assertEqual(self.events("chat_history.txt")[-1]["tags"], ["sleep", "rabbit"])

    def test_timestamped_heartbeat_only_does_not_call_model(self):
        self.save({**self.payload, "data": session_content("2026-10-21 12:01:00 | heart_beat")})
        self.answer.assert_not_called()
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], [])

    def test_tag_validation_and_limit_also_apply_without_summary(self):
        pairs = [f"concept-{index}" for index in range(6)]
        self.answer.return_value = {"content": json.dumps({
            "action": "no_sense", "tags": pairs,
        })}
        with patch.object(socket_handlers, "_dialogue_tag_lists", return_value=[]):
            self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], pairs[:6])

    def test_real_socket_ack_follows_saved_tags(self):
        app = Flask(__name__)
        socketio = SocketIO(app, async_mode="threading")
        socket_handlers.register_socket_handlers(socketio)
        client = socketio.test_client(app)
        try:
            response = client.emit("save_dialogue_history", self.payload, callback=True)
            self.assertEqual(response, "success")
            self.assertEqual(self.events("chat_history.txt")[-1]["tags"], ["sleep", "rabbit"])
            self.assertEqual(client.get_received(), [])
            loaded = client.emit("get_dialogue_history", self.payload, callback=True)
            self.assertEqual(loaded[0], "success")
            self.assertEqual(json.loads(loaded[1].splitlines()[-1])["tags"], ["sleep", "rabbit"])
        finally:
            client.disconnect()

    def test_concurrent_retry_does_not_duplicate_model_or_history(self):
        entered = threading.Event()
        release = threading.Event()
        response = self.answer.return_value
        results = []

        def answer(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test model wait timed out")
            return response

        self.answer.side_effect = answer
        first = threading.Thread(target=lambda: results.append(self.save(self.payload)))
        second = threading.Thread(target=lambda: results.append(self.save(self.payload)))
        first.start()
        try:
            self.assertTrue(entered.wait(5))
            second.start()
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results, ["success", "success"])
        self.answer.assert_called_once()
        self.assertEqual(len(self.events("chat_history.txt")), 3)

    def test_multiple_sessions_each_end_with_their_own_tags(self):
        payload = {**self.payload, "data": self.payload["data"] + session_content("Second", "2026-10-21 13:00:00")}
        self.assertEqual(self.save(payload), "success")
        events = self.events("chat_history.txt")
        self.assertEqual(len(events), 6)
        self.assertEqual(events[2]["tags"], ["sleep", "rabbit"])
        self.assertEqual(events[-1]["tags"], ["sleep", "rabbit"])
        self.assertEqual(self.answer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
