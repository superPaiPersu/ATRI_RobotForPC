import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from server import history_skill, test_history_summary as summary_tests
from client.pet_template.AI_tools import SocketAI
from client.pet_template.harness_bridge import live_dialogue_tags, run_history_search
from client.test_history_search import make_template_owner


class ConceptRegistryTests(unittest.TestCase):
    def test_empty_registry_and_concurrent_additions_do_not_lose_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(history_skill.dialogue_tag_lists(root), [])
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda i: history_skill.register_dialogue_tags(["兔子", f"概念{i}"], root), range(20)))
            labels = history_skill.dialogue_tag_lists(root)
            self.assertEqual(len(labels), 21)
            self.assertEqual(set(labels), {"兔子", *(f"概念{i}" for i in range(20))})
            before = (root / "data/dialogue_tags").read_bytes()
            history_skill.register_dialogue_tags(["兔子", "兔子"], root)
            self.assertEqual((root / "data/dialogue_tags").read_bytes(), before)

    def test_invalid_nested_tags_do_not_modify_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_skill.register_dialogue_tags(["日语"], root)
            for tags in ([["日语", "经历"]], ["a\nb"], [" "], [42]):
                with self.subTest(tags=tags), self.assertRaises(ValueError):
                    history_skill.register_dialogue_tags(tags, root)
            self.assertEqual(history_skill.dialogue_tag_lists(root), ["日语"])


class SavedConceptIntegrationTests(unittest.TestCase):
    setUp = summary_tests.AutomaticHistorySummaryTests.setUp
    tearDown = summary_tests.AutomaticHistorySummaryTests.tearDown
    events = summary_tests.AutomaticHistorySummaryTests.events
    def test_save_ack_updates_live_template_indexes_and_harness_vocabulary(self):
        self.tags.stop()
        history_skill.register_dialogue_tags(["sleep"], self.root)
        owner = make_template_owner([])
        owner.tags_lt = []
        with patch.object(SocketAI, "_connect"):
            client = SocketAI(username="test-user", pet_id="test-pet", history_search_callback=owner.tags_match)
        # Use real server handler and real client merge, without Qt or a remote model.
        def request(target, text, **kwargs):
            return self.save({**self.payload, "data": text, **kwargs})
        with patch.object(client, "wait_for_response", side_effect=request):
            self.assertEqual(client.save_dialogue_history(self.payload["data"]), "success")
            self.assertEqual(client.save_dialogue_history(self.payload["data"]), "success")
        self.assertEqual(len(owner.tags_dict), 1)
        self.assertEqual(owner.tags_dict[0], ["sleep", "rabbit"])
        self.assertEqual(owner.tags_lt, ["sleep", "rabbit"])
        self.assertEqual(live_dialogue_tags(owner.tags_match, []), owner.tags_lt)
        messages = run_history_search(owner.tags_match, ["rabbit"])["messages"]
        self.assertEqual(messages[1]["content"], "I often stay up late")
        self.answer.assert_called_once()
        self.assertEqual(history_skill.dialogue_tag_lists(self.root), ["sleep", "rabbit"])

    def test_reuse_rule_and_existing_table_are_in_model_request(self):
        self.save(self.payload)
        prompt = self.answer.call_args.args[3][0]["content"]
        self.assertIn('优先', prompt)
        self.assertIn('只有已有标签不能准确表达', prompt)
        self.assertIn('["sleep"]', prompt)

    def test_next_summary_sees_newly_registered_concepts(self):
        self.tags.stop()
        self.save(self.payload)
        from server.test_history_summary import session_content
        self.save({**self.payload, "data": session_content("rabbit", "2026-10-21 14:00:00")})
        self.assertIn('["sleep", "rabbit"]', self.answer.call_args.args[3][0]["content"])

    def test_excess_model_tags_are_bounded_without_discarding_session_index(self):
        self.answer.return_value = {"content": json.dumps({"action": "no_sense", "tags": [f"concept-{i}" for i in range(10)]})}
        self.save(self.payload)
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], [f"concept-{i}" for i in range(6)])

    def test_pair_output_is_rejected_without_losing_original_dialogue(self):
        self.answer.return_value = {"content": json.dumps({"action": "no_sense", "tags": [["sleep", "habit"]]})}
        self.assertEqual(self.save(self.payload), "success")
        self.assertEqual(self.events("chat_history.txt")[-1]["tags"], [])
        self.assertFalse((self.root / "data/dialogue_tags").exists())


if __name__ == "__main__":
    unittest.main()
