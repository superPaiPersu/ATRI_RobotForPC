import copy
import tempfile
import unittest
from pathlib import Path

from server.deduplicate_history import build_plan, deduplicate, split_sessions
from server.migrate_history_protocol import apply_plan, encode_sessions, read_rows, source_id


def session(ended="2026-10-21 12:10:00", started="2026-10-21 12:00:00", text="我经常熬夜"):
    rows = [
        {"type": "dialogue_session_start", "started_at": started, "marker": "PC", "messages": []},
        {"type": "dialogue_message", "messages": [{"role": "user", "content": text}]},
        {"type": "dialogue_message", "messages": [{"role": "assistant", "content": "wait", "action": "wait"}]},
        {"type": "dialogue_session_end", "ended_at": ended, "messages": [], "tags": []},
    ]
    rows[-1]["summary_source_id"] = source_id(rows)
    return rows


class HistoryDeduplicationTests(unittest.TestCase):
    def test_exact_and_retry_duplicates_keep_one_and_preserve_wait(self):
        first, retry = session(), session(ended="2026-10-21 12:11:00")
        retry[-1]["tags"] = [["睡眠", "习惯"]]
        originals = copy.deepcopy([first, first, retry])
        kept, aliases, removed = deduplicate(originals, "dialogue")
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 2)
        self.assertEqual(kept[0][2]["messages"][0]["content"], "wait")
        self.assertEqual(kept[0][-1]["tags"], [["睡眠", "习惯"]])
        self.assertEqual(kept[0][-1]["ended_at"], first[-1]["ended_at"])
        self.assertEqual(aliases[retry[-1]["summary_source_id"]], first[-1]["summary_source_id"])
        self.assertEqual(originals, [first, first, retry])

    def test_different_starts_or_different_content_are_preserved(self):
        sessions = [session(), session(started="2026-10-22 12:00:00"), session(text="我不熬夜")]
        self.assertEqual(deduplicate(sessions, "dialogue")[0], sessions)

    def test_repeated_messages_inside_one_session_are_preserved(self):
        rows = session()
        rows.insert(3, copy.deepcopy(rows[2]))
        self.assertEqual(deduplicate([rows, rows], "dialogue")[0], [rows])

    def test_unknown_time_does_not_merge_different_boundaries(self):
        first = session(started="未知")
        second = session(started="未知", ended="2026-10-22 12:00:00")
        self.assertEqual(len(deduplicate([first, second], "dialogue")[0]), 2)

    def test_relink_and_deduplicate_summary_after_source_duplicate_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "server/users_data/test/pet"
            folder.mkdir(parents=True)
            first, retry = session(), session(ended="2026-10-21 12:11:00")
            (folder / "chat_history.txt").write_bytes(encode_sessions([first, retry]))
            summaries = []
            for source in (first, retry):
                summary = copy.deepcopy(source)
                for row in summary:
                    row["type"] = row["type"].replace("dialogue_", "summary_")
                summary[-1]["action"] = "summary"
                summaries.append(summary)
            original = encode_sessions(summaries)
            (folder / "summary_history.txt").write_bytes(original)
            plan, report = build_plan(root)
            backup = apply_plan(root, plan)
            self.assertEqual([r["removed"] for r in report], [1, 1])
            saved = split_sessions(read_rows((folder / "summary_history.txt").read_bytes()), "summary")
            self.assertEqual(saved[0][-1]["summary_source_id"], first[-1]["summary_source_id"])
            self.assertEqual((backup / "server/users_data/test/pet/summary_history.txt").read_bytes(), original)
            self.assertEqual(build_plan(root)[0], [])

    def test_same_summary_for_distinct_sources_is_not_removed(self):
        rows = session()
        for row in rows:
            row["type"] = row["type"].replace("dialogue_", "summary_")
        second = copy.deepcopy(rows)
        second[-1]["summary_source_id"] = "different-source"
        self.assertEqual(len(deduplicate([rows, second], "summary")[0]), 2)

    def test_broken_sessions_fail_without_deleting_anything(self):
        with self.assertRaises(ValueError):
            split_sessions(session()[:-1], "dialogue")
        with self.assertRaises(ValueError):
            split_sessions(session()[1:], "dialogue")


if __name__ == "__main__":
    unittest.main()
