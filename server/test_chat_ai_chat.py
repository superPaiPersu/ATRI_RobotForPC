import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server import chat_ai


def text_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, tool_calls=None),
        finish_reason=None,
    )])


def tool_chunk():
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            index=0,
            id="call-1",
            type="function",
            function=SimpleNamespace(name="bocha_web_search", arguments='{"query":"weather"}'),
        )]),
        finish_reason="tool_calls",
    )])


class ChatCompletionsTests(unittest.TestCase):
    def test_gpt_stream_emits_existing_client_protocol(self):
        client = Mock()
        client.chat.completions.create.return_value = iter([
            text_chunk('{"type":"params","action":"reply","mood":"normal"}'),
            text_chunk('{"type":"segment","seq":1,"jp":"はい","cn":"好的"}'),
            text_chunk('{"type":"end"}'),
        ])
        chunks = []
        with patch.object(chat_ai, "_create_openai_client", return_value=client):
            result = chat_ai.get_answer_gpt("https://example.invalid/v1", "test-key", [
                {"role": "user", "content": "你好"},
            ], on_chunk=lambda flag, payload: chunks.append((flag, payload)))

        self.assertEqual(json.loads(result["content"])["cn"], "好的")
        self.assertEqual([flag for flag, _ in chunks], ["params", "segment", "end"])
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertTrue(request["stream"])
        self.assertIn("bocha_web_search", [tool["function"]["name"] for tool in request["tools"]])

    def test_gpt_tool_result_uses_chat_messages(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            iter([tool_chunk()]),
            iter([text_chunk('{"action":"reply","cn":"晴天","jp":"晴れ"}')]),
        ]
        with patch.object(chat_ai, "_create_openai_client", return_value=client), patch.object(
            chat_ai, "_run_chat_completion_tool_call", return_value='{"result":"sunny"}'
        ):
            result = chat_ai.get_answer_gpt("https://example.invalid/v1", "test-key", [
                {"role": "user", "content": "天气"},
            ])

        self.assertEqual(json.loads(result["content"])["cn"], "晴天")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[-1], {
            "role": "tool", "tool_call_id": "call-1", "content": '{"result":"sunny"}',
        })

    def test_gpt_image_uses_chat_content_parts(self):
        with patch.object(chat_ai, "_image_reference_to_data_url", return_value=(
            "data:image/png;base64,AAAA", "image/png", None,
        )):
            parts = chat_ai._serialize_content_for_model([
                "What is this?",
                {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
            ], "gpt", True)

        self.assertEqual(parts[0], {"type": "text", "text": "What is this?"})
        self.assertEqual(parts[1], {
            "type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"},
        })

    def test_translation_uses_chat_completion(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content="你好")),
        ])
        with patch.object(chat_ai, "_create_translation_client", return_value=client):
            self.assertEqual(chat_ai.translate_text("こんにちは"), "你好")

        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][-1]["content"], "こんにちは")
        self.assertEqual(request["max_completion_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
