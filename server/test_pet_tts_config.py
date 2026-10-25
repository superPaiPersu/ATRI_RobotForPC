import json
import tempfile
import unittest
from pathlib import Path

from server.tts_backend import apply_pet_tts_config


class PetTtsConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.pet = Path(temp.name)
        self.config = self.pet / 'config'
        self.refs = self.config / 'tts_refs'
        self.refs.mkdir(parents=True)
        self.original = self.config / 'voice_refs/正常谈话_话题平常.ogg'
        self.target = self.refs / 'ref001_こんにちは。.wav'
        self.target.write_bytes(b'reference')
        self.profile = {
            'backend': 'api_v2', 'prompt_lang': 'ja',
            'prompt_text_mode': 'transcript', 'reference_dir': 'tts_refs',
            'reference_aliases': {'正常谈话_话题平常': self.target.stem},
            'reference_prompts': {'正常谈话_话题平常': 'こんにちは。'},
        }

    def save(self):
        (self.config / 'tts.json').write_text(json.dumps(self.profile, ensure_ascii=False), encoding='utf-8')

    def apply(self, payload=None):
        return apply_pet_tts_config(self.pet, self.original.name, self.original, payload or {})

    def test_without_profile_preserves_existing_behavior(self):
        payload = {'prompt_text': 'existing'}
        self.assertEqual(self.apply(payload), (self.original, payload))

    def test_transcript_overrides_emotion_description_without_mutation(self):
        self.save()
        payload = {'prompt_text': '话题平常', 'text': 'hello'}
        path, result = self.apply(payload)
        self.assertEqual(path, self.target)
        self.assertEqual(result['prompt_text'], 'こんにちは。')
        self.assertEqual(result['prompt_lang'], 'ja')
        self.assertEqual(result['tts_backend'], 'api_v2')
        self.assertEqual(payload['prompt_text'], '话题平常')

    def test_missing_transcript_does_not_fall_back_to_filename(self):
        self.profile['reference_prompts'] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, 'Missing reference transcript'):
            self.apply({'prompt_text': '话题平常'})

    def test_reference_directory_cannot_escape_config(self):
        self.profile['reference_dir'] = '../outside'
        self.save()
        with self.assertRaisesRegex(ValueError, 'cannot leave config'):
            self.apply()

    def test_reference_alias_cannot_escape_directory(self):
        self.profile['reference_aliases']['正常谈话_话题平常'] = '../outside'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid pet reference alias'):
            self.apply()

    def test_explicit_backend_is_preserved(self):
        self.save()
        self.assertEqual(self.apply({'tts_backend': 'new_api'})[1]['tts_backend'], 'new_api')

    def test_filename_mode_remains_supported(self):
        self.profile['prompt_text_mode'] = 'filename'
        self.save()
        self.assertEqual(self.apply()[1]['prompt_text'], 'こんにちは。')

    def test_missing_prepared_audio_fails_early(self):
        self.save()
        self.target.unlink()
        with self.assertRaisesRegex(ValueError, 'Prepared reference audio not found'):
            self.apply()


if __name__ == '__main__':
    unittest.main()
