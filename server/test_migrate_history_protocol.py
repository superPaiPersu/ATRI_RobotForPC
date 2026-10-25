import copy
import json
import tempfile
import unittest
from pathlib import Path

from server.migrate_history_protocol import (
    UNKNOWN_TIME, apply_plan, build_plan, encode_sessions, normalize_sessions, read_rows, source_id,
)


def message(content, role="user", kind="dialogue", **extra):
    return {"type": f"{kind}_message", "messages": [{"role": role, "content": content, **extra}]}


def session(kind="dialogue", started_at="2026-10-21 12:00:00"):
    return [
        {"type": f"{kind}_session_start", "started_at": started_at, "marker": "PC", "messages": []},
        message("wait", "assistant", kind, action="wait"),
        {"type": f"{kind}_session_end", "ended_at": "2026-10-21 12:10:00", "messages": []},
    ]


class HistoryMigrationTests(unittest.TestCase):
    def test_current_hash_matches_server_protocol_and_wait_is_unchanged(self):
        from server.socket_handlers import _saved_dialogue_sessions
        rows = session()
        before = copy.deepcopy(rows)
        migrated = normalize_sessions(rows, "dialogue")
        expected = _saved_dialogue_sessions(encode_sessions([rows]).decode("utf-8"))
        self.assertEqual(migrated[0][-1]["summary_source_id"], expected[0][-1]["summary_source_id"])
        self.assertEqual(migrated[0][1], rows[1])
        self.assertEqual(rows, before)
        self.assertEqual(migrated[0][-1]["tags"], [])
        self.assertEqual(normalize_sessions(read_rows(encode_sessions(migrated)), "dialogue"), migrated)

    def test_old_text_boundaries_are_recovered_without_changing_speech(self):
        start = message("[系统消息]:一段历史对话开始于 2026-05-24 01:29:45，记录标记：唤醒model。")
        end = message("[系统消息]:上一段历史对话结束于 2026-05-24 01:29:55。")
        speech = message("wait", "assistant", action="wait")
        result = normalize_sessions([start, speech, end], "dialogue")[0]
        self.assertEqual(result[0]["started_at"], "2026-05-24 01:29:45")
        self.assertEqual(result[0]["marker"], "唤醒model")
        self.assertEqual(result[-1]["ended_at"], "2026-05-24 01:29:55")
        self.assertEqual(result[0]["legacy_event"], start)
        self.assertEqual(result[-1]["legacy_event"], end)
        self.assertEqual(result[1], speech)

    def test_legacy_missing_end_is_explicitly_unknown(self):
        rows = [message("[系统消息]:一段历史对话开始于 2026-05-24 01:29:45，记录标记：PC。"),
                message("第一段"),
                message("[系统消息]:一段历史对话开始于 2026-05-25 01:29:45，记录标记：PC。"),
                message("第二段")]
        result = normalize_sessions(rows, "dialogue")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][-1]["ended_at"], UNKNOWN_TIME)
        self.assertEqual(result[1][-1]["ended_at"], UNKNOWN_TIME)
        self.assertEqual(result[0][1], rows[1])
        self.assertEqual(result[1][1], rows[3])

    def test_unbounded_records_and_empty_files(self):
        result = normalize_sessions([message("旧记录")], "dialogue")
        self.assertEqual(result[0][0]["started_at"], UNKNOWN_TIME)
        self.assertEqual(result[0][-1]["ended_at"], UNKNOWN_TIME)
        self.assertEqual(normalize_sessions([], "summary"), [])

    def test_summary_links_only_unique_known_start_and_preserves_old_text(self):
        dialogues = normalize_sessions(session(), "dialogue")
        summary = session("summary")
        result = normalize_sessions(summary, "summary", dialogues)[0]
        self.assertEqual(result[-1]["summary_source_id"], dialogues[0][-1]["summary_source_id"])
        self.assertEqual(result[-1]["action"], "summary")
        self.assertEqual(result[1], summary[1])
        for candidates in ([], dialogues * 2):
            result = normalize_sessions(summary, "summary", candidates)[0]
            self.assertTrue(result[-1]["legacy_unlinked"])
            self.assertNotEqual(result[-1]["summary_source_id"], dialogues[0][-1]["summary_source_id"])
            self.assertEqual(normalize_sessions(result, "summary", candidates), [result])

    def test_broken_native_records_fail_without_guessing(self):
        with self.assertRaises(ValueError):
            normalize_sessions(session()[:-1], "dialogue")
        with self.assertRaises(ValueError):
            normalize_sessions(session()[2:], "dialogue")
        with self.assertRaises(ValueError):
            read_rows(b'{"type":')

    def test_backup_preserves_exact_bytes_and_second_run_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "server/users_data/test/pet/chat_history.txt"
            path.parent.mkdir(parents=True)
            original = b"\xef\xbb\xbf" + encode_sessions([session()]).replace(b"\n", b"\r\n")
            path.write_bytes(original)
            plan = build_plan(root)
            backup = apply_plan(root, plan)
            self.assertEqual((backup / path.relative_to(root)).read_bytes(), original)
            self.assertEqual(len(json.loads((backup / "manifest.json").read_text(encoding="utf-8"))), 1)
            self.assertEqual(build_plan(root), [])
            self.assertEqual(read_rows(path.read_bytes())[1], session()[1])

    def test_concurrent_change_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "server/users_data/test/pet/chat_history.txt"
            path.parent.mkdir(parents=True)
            path.write_bytes(encode_sessions([session()]))
            plan = build_plan(root)
            path.write_bytes(b"changed during migration")
            with self.assertRaises(RuntimeError):
                apply_plan(root, plan)
            self.assertEqual(path.read_bytes(), b"changed during migration")


if __name__ == "__main__":
    unittest.main()
