"""Exercise the actual template method without starting Qt or importing main."""
import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

from client.pet_template.harness_bridge import run_history_search


TAGS = ["睡眠"]


def make_template_owner(pairs):
    path = Path(__file__).resolve().parent / "pet_template/main.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    widget = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GLWidget")
    function = next(node for node in widget.body if isinstance(node, ast.FunctionDef) and node.name == "tags_match")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    owner = SimpleNamespace(tags_dict=dict(enumerate(pairs)), tags_index_dict={})
    for index in owner.tags_dict:
        owner.tags_index_dict[index] = [
            {"role": "developer", "content": f"session {index} start"},
            {"role": "user", "content": f"memory {index}"},
            {"role": "assistant", "content": "wait", "action": "wait"},
            {"role": "developer", "content": f"session {index} end"},
        ]
    owner.tags_match = MethodType(namespace["tags_match"], owner)
    return owner


class HistorySearchAdapterTests(unittest.TestCase):
    def test_whole_concepts_do_not_match_character_order_or_prefixes(self):
        owner = make_template_owner([["日语"], ["语日"], ["日语助词"], ["宫保鸡丁"]])
        self.assertEqual(run_history_search(owner.tags_match, ["日语"])["messages"], owner.tags_index_dict[0])
        self.assertFalse(run_history_search(owner.tags_match, ["日"])["matched"])

    def test_actual_template_zero_one_four_five_and_many_sessions(self):
        for count in (0, 1, 4, 5, 8):
            with self.subTest(count=count):
                owner = make_template_owner([TAGS] * count)
                result = run_history_search(owner.tags_match, TAGS)
                expected = [message for index in range(min(count, 5)) for message in owner.tags_index_dict[index]]
                self.assertEqual(result, {"messages": expected, "matched": bool(expected)})

    def test_zero_scores_are_excluded_even_when_template_returns_ids(self):
        for count in (1, 5, 8):
            with self.subTest(count=count):
                owner = make_template_owner([["饮食"]] * count)
                self.assertEqual(run_history_search(owner.tags_match, TAGS), {"messages": [], "matched": False})

    def test_ranked_sessions_expand_in_order_and_read_live_history(self):
        second = ["饮食"]
        owner = make_template_owner([[], TAGS, TAGS + second, [], second, []])
        result = run_history_search(owner.tags_match, TAGS + second)
        expected = owner.tags_index_dict[2] + owner.tags_index_dict[1] + owner.tags_index_dict[4]
        self.assertEqual(result["messages"], expected)
        owner.tags_index_dict[2].append({"role": "assistant", "content": "new memory"})
        self.assertEqual(run_history_search(owner.tags_match, TAGS + second)["messages"][4]["content"], "new memory")

    def test_indexed_parser_empty_items_are_skipped_without_dropping_wait(self):
        owner = make_template_owner([TAGS])
        owner.tags_index_dict[0].insert(2, None)
        result = run_history_search(owner.tags_match, TAGS)
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["messages"][2]["action"], "wait")

    def test_unknown_session_and_unbound_ids_are_errors(self):
        with self.assertRaises(ValueError):
            run_history_search(lambda tags: [1], TAGS)
        owner = make_template_owner([TAGS] * 5)
        def bad(self, tags):
            return [99]
        with self.assertRaises(ValueError):
            run_history_search(MethodType(bad, owner), TAGS)

    def test_callback_index_error_is_not_hidden(self):
        owner = make_template_owner([TAGS])
        def tags_match(self, tags):
            raise IndexError("callback bug")
        with self.assertRaises(IndexError):
            run_history_search(MethodType(tags_match, owner), TAGS)

    def test_duplicate_session_ids_are_not_returned_twice(self):
        owner = make_template_owner([TAGS] * 5)
        def repeated(self, tags):
            return [1, 1, 2]
        result = run_history_search(MethodType(repeated, owner), TAGS)
        self.assertEqual(result["messages"], owner.tags_index_dict[1] + owner.tags_index_dict[2])


if __name__ == "__main__":
    unittest.main()
