import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from client import bocha_mcp_server, harness_service
from client.pet_template.harness_bridge import PetHarnessBridge, call_pet_bridge


class HarnessBridgeTests(unittest.TestCase):
    def test_template_session_ids_resolve_over_real_loopback(self):
        from client.test_history_search import make_template_owner
        for count in (1, 6):
            with self.subTest(count=count):
                owner = make_template_owner([["睡眠"]] + [[]] * (count - 1))
                self.bridge.search_callback = owner.tags_match
                result = self.call("search_dialogue_history", tags=["睡眠"])
                self.assertEqual(result, {"messages": owner.tags_index_dict[0], "matched": True})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.messages = [{"role": "user", "content": "昨晚熬夜了"}]
        self.tags = ["睡眠"]
        self.search = Mock(side_effect=lambda tags: list(self.messages))
        self.bridge = PetHarnessBridge(
            self.search, lambda: self.tags, username="test-user", pet_id="pet",
            registry_dir=self.root,
        ).start()

    def tearDown(self):
        self.bridge.close()
        self.temporary.cleanup()

    def call(self, method, **kwargs):
        return call_pet_bridge(self.root, "test-user", "pet", method, **kwargs)

    def test_live_history_and_vocabulary_over_real_loopback(self):
        self.assertEqual(self.call("get_dialogue_tags"), self.tags)
        result = self.call("search_dialogue_history", tags=["睡眠"])
        self.assertEqual(result["messages"], self.messages)
        self.search.assert_called_once_with(["睡眠"])
        self.messages.append({"role": "assistant", "content": "早点休息"})
        self.assertEqual(len(self.call("search_dialogue_history", tags=["睡眠"])["messages"]), 2)

    def test_no_match_and_callback_error_are_distinct(self):
        self.search.side_effect = KeyError(None)
        self.assertEqual(self.call("search_dialogue_history", tags=["睡眠"]), {"messages": [], "matched": False})
        self.search.side_effect = RuntimeError("private detail")
        with self.assertRaisesRegex(RuntimeError, "RuntimeError") as failure:
            self.call("search_dialogue_history", tags=["睡眠"])
        self.assertNotIn("private detail", str(failure.exception))

    def test_authentication_isolation_validation_and_cleanup(self):
        with self.assertRaises(RuntimeError):
            call_pet_bridge(self.root, "other-user", "pet", "get_dialogue_tags")
        with self.assertRaises(ValueError):
            self.call("search_dialogue_history", tags=[["睡眠", "经历"]])
        descriptor = json.loads(self.bridge.path.read_text(encoding="utf-8"))
        self.bridge.path.write_text(json.dumps({**descriptor, "token": "wrong-token"}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.call("get_dialogue_tags")
        self.bridge.path.write_text(json.dumps(descriptor), encoding="utf-8")
        self.bridge.close()
        self.assertFalse(self.bridge.path.exists())

    def test_mcp_calls_local_bridge_without_server_socket(self):
        unused_server = Mock()
        with patch.object(bocha_mcp_server, "USERNAME", "test-user"), patch.object(
            bocha_mcp_server, "PET_ID", "pet"
        ), patch.object(bocha_mcp_server, "PET_BRIDGE_REGISTRY", str(self.root)):
            result = bocha_mcp_server._handle_request({
                "id": 1, "method": "tools/call", "params": {
                    "name": "search_dialogue_history", "arguments": {"tags": ["睡眠"]},
                },
            }, unused_server)
        self.assertEqual(json.loads(result["result"]["content"][0]["text"])["messages"], self.messages)
        self.assertEqual(unused_server.mock_calls, [])

    def test_separate_python_process_can_query_pet_with_proxy_enabled(self):
        script = (
            "import json,sys; from client.pet_template.harness_bridge import call_pet_bridge; "
            "print(json.dumps(call_pet_bridge(sys.argv[1], 'test-user', 'pet', 'get_dialogue_tags')))"
        )
        environment = os.environ.copy()
        environment.update({"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""})
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root)],
            cwd=Path(__file__).resolve().parent.parent, env=environment,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), self.tags)

    def test_harness_workspace_contains_bridge_and_discovery_config(self):
        service = harness_service.HarnessService(Path(__file__).resolve().parent, "test-user")
        service.data_root = self.root / "harness"
        with patch.object(harness_service, "DeepSeekHarness"):
            _, workspace = service._ensure_harness("pet", "https://example.invalid/v1", "test-key", "test-model")
        self.assertTrue((workspace / "harness_bridge.py").is_file())
        config = json.loads((workspace / "bocha_mcp_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["pet_id"], "pet")
        self.assertEqual(config["pet_bridge_registry"], str(service.base_dir / "client_data" / "harness_bridge"))

    def test_skill_is_installed_for_each_pet_and_refreshed_on_runtime_reuse(self):
        service = harness_service.HarnessService(Path(__file__).resolve().parent, "test-user")
        service.data_root = self.root / "harness"
        original = (service.base_dir / "skills" / "dialogue-memory" / "SKILL.md").read_bytes()
        with patch.object(harness_service, "DeepSeekHarness") as constructor:
            runtime, workspace = service._ensure_harness("pet", "https://example.invalid/v1", "test-key", "test-model")
            installed = workspace.parent / "home" / "skills" / "dialogue-memory" / "SKILL.md"
            self.assertEqual(installed.read_bytes(), original)
            unchanged_mtime = installed.stat().st_mtime_ns
            reused, _ = service._ensure_harness("pet", "https://example.invalid/v1", "test-key", "test-model")
            self.assertIs(reused, runtime)
            self.assertEqual(installed.stat().st_mtime_ns, unchanged_mtime)
            installed.write_text("stale skill", encoding="utf-8")
            service._ensure_harness("pet", "https://example.invalid/v1", "test-key", "test-model")
            self.assertEqual(installed.read_bytes(), original)
            constructor.assert_called_once()
            _, other_workspace = service._ensure_harness("other-pet", "https://example.invalid/v1", "test-key", "test-model")
            other_skill = other_workspace.parent / "home" / "skills" / "dialogue-memory" / "SKILL.md"
            self.assertEqual(other_skill.read_bytes(), original)
            self.assertNotEqual(other_skill, installed)


if __name__ == "__main__":
    unittest.main()
