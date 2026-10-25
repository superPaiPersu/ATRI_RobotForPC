import base64
import copy
import io
import json
import pickle
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests

from server import socket_handlers, tts_backend, tts_config
from server.test_tts_prompt import FakeResponse, FakeSocketIO


def make_wav(channels=1):
    samples = np.array([0, 1000, -1000, 2000] * channels, dtype=np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(32000)
        audio.writeframes(samples.tobytes())
    return output.getvalue(), samples


class TtsBackendTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "text": "hello", "text_lang": "jp", "prompt_lang": "jp",
            "prompt_text": "reference", "ref_audio_path": "normal_reference.wav",
            "model_name": "old-trained-voice", "sample_steps": 32,
        }

    def test_new_api_preserves_payload_and_legacy_endpoint(self):
        with patch.object(tts_config, "TTS_BACKEND", "new_api"):
            backend, url, payload, timeout = tts_backend.prepare_tts_request(self.payload)
        self.assertEqual((backend, url, timeout), ("new_api", tts_config.NEW_API_URL, 60))
        self.assertEqual(payload, self.payload)

    def test_api_v2_filters_model_and_forces_complete_wav(self):
        self.payload.update(tts_backend="api_v2", streaming_mode=True, media_type="ogg",
                            api_url="http://untrusted.invalid", weights_path="trained.pth")
        original = copy.deepcopy(self.payload)
        backend, url, payload, timeout = tts_backend.prepare_tts_request(self.payload)
        self.assertEqual((backend, url, timeout), ("api_v2", tts_config.API_V2_URL, 120))
        self.assertEqual(self.payload, original)
        for field in ("model_name", "tts_backend", "api_url", "weights_path"):
            self.assertNotIn(field, payload)
        self.assertEqual(payload["text_lang"], "ja")
        self.assertEqual(payload["prompt_lang"], "ja")
        self.assertEqual(payload["sample_steps"], 32)
        self.assertEqual(payload["media_type"], "wav")
        self.assertIs(payload["streaming_mode"], False)

    def test_config_selects_api_v2_without_client_changes(self):
        with patch.object(tts_config, "TTS_BACKEND", "api_v2"):
            self.assertEqual(tts_backend.prepare_tts_request(self.payload)[0], "api_v2")

    def test_explicit_new_api_overrides_server_default(self):
        self.payload["tts_backend"] = "new_api"
        with patch.object(tts_config, "TTS_BACKEND", "api_v2"):
            backend, _, payload, _ = tts_backend.prepare_tts_request(self.payload)
        self.assertEqual(backend, "new_api")
        self.assertEqual(payload["model_name"], "old-trained-voice")
        self.assertNotIn("tts_backend", payload)

    def test_unknown_backend_is_rejected(self):
        self.payload["tts_backend"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "Unsupported tts_backend"):
            tts_backend.prepare_tts_request(self.payload)

    def test_required_api_v2_fields_are_validated(self):
        for field in ("text", "text_lang", "prompt_lang", "ref_audio_path", "prompt_text"):
            for bad_value in (None, "", "  ", 12):
                with self.subTest(field=field, value=bad_value):
                    payload = {**self.payload, "tts_backend": "api_v2", field: bad_value}
                    with self.assertRaises(ValueError):
                        tts_backend.prepare_tts_request(payload)

    def test_wav_to_pc_pickle_preserves_samples_and_channels(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                content, samples = make_wav(channels)
                result = pickle.loads(tts_backend.format_api_v2_response(content))
                self.assertEqual(result[-1][0], 32000)
                expected = samples if channels == 1 else samples.reshape((-1, channels))
                np.testing.assert_array_equal(result[-1][1], expected)

    def test_mobile_wav_and_base64_are_preserved(self):
        content, _ = make_wav()
        self.assertEqual(tts_backend.format_api_v2_response(content, "wav"), content)
        result = tts_backend.format_api_v2_response(content, "base64")
        self.assertIs(result["ok"], True)
        self.assertEqual(base64.b64decode(result["audio_base64"]), content)

    def test_invalid_upstream_responses_are_rejected(self):
        for content in (b"", b"<html>error</html>", b'{"error":"oops"}', b"RIFF0000WAVE"):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    tts_backend.format_api_v2_response(content)


class TtsBackendEventTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.reference = self.root / "atri" / "config" / "voice_refs" / "normal_reference_prompt.wav"
        self.reference.parent.mkdir(parents=True)
        self.reference.write_bytes(b"reference")
        socket = FakeSocketIO()
        socket_handlers.register_socket_handlers(socket)
        self.handler = socket.handlers["voice_produce"]
        for attribute, value in (("SERVER_PETS_DIR", self.root),
                                 ("request", SimpleNamespace(sid="tts-test"))):
            patcher = patch.object(socket_handlers, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(socket_handlers, "_authenticated_username", return_value=("user", ""))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = {"data": {
            "pet_id": "atri", "ref_audio_name": "normal.wav", "tts_backend": "api_v2",
            "text": "hello", "text_lang": "ja", "prompt_lang": "ja",
            "prompt_text": "", "model_name": "old-model",
        }}

    def test_api_v2_event_resolves_mood_and_preserves_pc_contract(self):
        response = FakeResponse()
        response.content, samples = make_wav()
        original = copy.deepcopy(self.payload)
        with patch.object(socket_handlers.requests, "post", return_value=response) as post:
            result = self.handler(self.payload)
        request = post.call_args.kwargs
        self.assertEqual(request["url"], tts_config.API_V2_URL)
        self.assertEqual(request["json"]["ref_audio_path"], str(self.reference))
        self.assertEqual(request["json"]["prompt_text"], "reference_prompt")
        self.assertNotIn("model_name", request["json"])
        self.assertEqual(self.payload, original)
        np.testing.assert_array_equal(pickle.loads(result)[-1][1], samples)

    def test_new_api_mood_fallback_regression(self):
        self.payload["data"]["tts_backend"] = "new_api"
        response = FakeResponse()
        with patch.object(socket_handlers.requests, "post", return_value=response) as post:
            result = self.handler(self.payload)
        self.assertEqual(result, response.content)
        self.assertEqual(post.call_args.kwargs["url"], tts_config.NEW_API_URL)
        self.assertEqual(post.call_args.kwargs["json"]["prompt_text"], "reference_prompt")
        self.assertEqual(post.call_args.kwargs["json"]["model_name"], "old-model")

    def test_api_v2_event_supports_mobile_formats(self):
        response = FakeResponse()
        response.content, _ = make_wav()
        for format_name in ("wav", "base64"):
            with self.subTest(format_name=format_name):
                self.payload["data"]["return_format"] = format_name
                with patch.object(socket_handlers.requests, "post", return_value=response):
                    result = self.handler(self.payload)
                actual = base64.b64decode(result["audio_base64"]) if format_name == "base64" else result
                self.assertEqual(actual, response.content)

    def test_missing_reference_does_not_call_upstream(self):
        self.reference.unlink()
        with patch.object(socket_handlers.requests, "post") as post:
            result = self.handler(self.payload)
        self.assertFalse(result["ok"])
        self.assertIn("reference audio not found", result["error"])
        post.assert_not_called()

    def test_pet_profile_sends_prepared_audio_and_real_transcript(self):
        config = self.root / 'atri/config'
        prepared = config / 'tts_refs/ref001_実際の音声。.wav'
        prepared.parent.mkdir()
        prepared.write_bytes(b'reference')
        (config / 'tts.json').write_text(json.dumps({
            'backend': 'api_v2', 'prompt_lang': 'ja',
            'prompt_text_mode': 'transcript', 'reference_dir': 'tts_refs',
            'reference_aliases': {'normal': prepared.stem},
            'reference_prompts': {'normal': '実際の音声。'},
        }, ensure_ascii=False), encoding='utf-8')
        response = FakeResponse()
        response.content, _ = make_wav()
        with patch.object(socket_handlers.requests, 'post', return_value=response) as post:
            result = self.handler(self.payload)
        self.assertIsInstance(result, bytes)
        self.assertEqual(post.call_args.kwargs['json']['ref_audio_path'], str(prepared))
        self.assertEqual(post.call_args.kwargs['json']['prompt_text'], '実際の音声。')

    def test_pet_profile_missing_transcript_does_not_call_upstream(self):
        (self.root / 'atri/config/tts.json').write_text(
            json.dumps({'prompt_text_mode': 'transcript', 'reference_prompts': {}}), encoding='utf-8')
        with patch.object(socket_handlers.requests, 'post') as post:
            result = self.handler(self.payload)
        self.assertFalse(result['ok'])
        self.assertIn('Missing reference transcript', result['error'])
        post.assert_not_called()

    def test_missing_prompt_does_not_call_upstream(self):
        self.reference.rename(self.reference.with_name("normal.wav"))
        with patch.object(socket_handlers.requests, "post") as post:
            result = self.handler(self.payload)
        self.assertFalse(result["ok"])
        self.assertIn("prompt_text", result["error"])
        post.assert_not_called()

    def test_upstream_timeout_returns_structured_error(self):
        with patch.object(socket_handlers.requests, "post", side_effect=requests.Timeout("test timeout")):
            result = self.handler(self.payload)
        self.assertFalse(result["ok"])
        self.assertIn("test timeout", result["error"])

    def test_html_response_returns_structured_error(self):
        response = FakeResponse()
        response.content = b"<html>not audio</html>"
        with patch.object(socket_handlers.requests, "post", return_value=response):
            result = self.handler(self.payload)
        self.assertFalse(result["ok"])
        self.assertIn("did not return WAV", result["error"])


if __name__ == "__main__":
    unittest.main()
