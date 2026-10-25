import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server import backfill_history_tags as backfill
from server.migrate_history_protocol import encode_sessions, read_rows


TOPICS = ["睡眠", "饮食"]


def session():
    return [
        {"type": "dialogue_session_start", "started_at": "2026-10-21 01:00:00", "messages": []},
        {"type": "dialogue_message", "messages": [{"role": "user", "content": "我经常熬夜"}]},
        {"type": "dialogue_message", "messages": [{"role": "assistant", "content": "wait", "action": "wait"}]},
        {"type": "dialogue_session_end", "ended_at": "2026-10-21 01:10:00", "tags": [],
         "summary_source_id": "test-source", "messages": []},
    ]


class BackfillHistoryTagTests(unittest.TestCase):
    def test_complete_payload_includes_wait_and_source_time(self):
        rows = session()
        payload = backfill.session_payload(rows, "dialogue")
        self.assertEqual(payload["dialogue"], rows[1]["messages"] + rows[2]["messages"])
        self.assertEqual(payload["started_at"], rows[0]["started_at"])

    def test_duplicate_client_and_server_sessions_generate_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "server/users_data/test/pet/chat_history.txt",
                     root / "client/client_data/pets/pet/config/dialogue_history.txt"]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_bytes(encode_sessions([session()]))
            files, jobs = backfill.collect_jobs(root)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(files[0]["identifiers"], files[1]["identifiers"])
            results = {key: {"tags": ["睡眠"], "empty_reason": ""} for key in jobs}
            plan, report = backfill.build_tag_plan(files, results)
            self.assertEqual(len(plan), 2)
            for entry in plan:
                before, after = read_rows(entry["before"]), read_rows(entry["after"])
                self.assertEqual(after[-1]["tags"], ["睡眠"])
                after[-1]["tags"] = before[-1]["tags"]
                self.assertEqual(after, before)
            self.assertEqual([row["tagged"] for row in report], [1, 1])

    def test_flat_tags_allow_new_concepts_but_reject_invalid_structure(self):
        valid = {"id": "a", "tags": ["睡眠", "兔子"], "empty_reason": ""}
        self.assertEqual(backfill.validate_results({"results": [valid]}, {"a"}, TOPICS)["a"]["tags"], valid["tags"])
        bad_results = [[], [valid, valid], [{**valid, "id": "b"}],
                       [{**valid, "tags": [["睡眠", "习惯"]]}],
                       [{**valid, "tags": ["  "]}],
                       [{**valid, "tags": valid["tags"] * 2}],
                       [{**valid, "tags": []}], [{**valid, "tags": "sleep"}]]
        for rows in bad_results:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                backfill.validate_results({"results": rows}, {"a"}, TOPICS)
        empty = {**valid, "tags": [], "empty_reason": "仅测试连接"}
        self.assertEqual(backfill.validate_results({"results": [empty]}, {"a"}, TOPICS)["a"]["tags"], [])

    def test_batching_never_truncates_history(self):
        jobs = [{"id": str(i), "dialogue": "x" * (i + 1)} for i in range(20)]
        batches = list(backfill.batches(jobs, limit=3, char_limit=120))
        self.assertEqual([job for batch in batches for job in batch], jobs)
        self.assertTrue(all(len(batch) <= 3 for batch in batches))

    def test_linked_summary_uses_source_dialogue_tags(self):
        rows = session()
        summary = [{**row, "type": row["type"].replace("dialogue_", "summary_")} for row in rows]
        files = [{"path": Path("chat_history.txt"), "before": encode_sessions([rows]),
                  "sessions": [rows], "identifiers": ["raw"]},
                 {"path": Path("summary_history.txt"), "before": encode_sessions([summary]),
                  "sessions": [summary], "identifiers": ["summary"]}]
        results = {"raw": {"tags": ["睡眠"]}, "summary": {"tags": ["饮食"]}}
        backfill.build_tag_plan(files, results)
        self.assertEqual(summary[-1]["tags"], rows[-1]["tags"])

    def test_invalid_model_output_retries_once_not_forever(self):
        valid = {"a": {"tags": ["睡眠"], "empty_reason": ""}}
        with patch.object(backfill, "_request_tags_once", side_effect=[ValueError("bad"), valid]) as call:
            self.assertEqual(backfill.request_tags(None, "test", [], TOPICS), valid)
            self.assertEqual(call.call_count, 2)
        with patch.object(backfill, "_request_tags_once", side_effect=ValueError("bad")) as call:
            with self.assertRaises(ValueError):
                backfill.request_tags(None, "test", [], TOPICS)
            self.assertEqual(call.call_count, 2)

    def test_wire_ids_are_short_and_restored_to_content_hashes(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"results": [{"id": "s0", "tags": ["睡眠"], "empty_reason": ""}]})))])
        original_id = "a" * 64
        result = backfill._request_tags_once(client, "test", [{"id": original_id, "dialogue": []}], TOPICS)
        self.assertEqual(result, {original_id: {"tags": ["睡眠"], "empty_reason": ""}})
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn('"id": "s0"', prompt)
        self.assertNotIn(original_id, prompt)


if __name__ == "__main__":
    unittest.main()
