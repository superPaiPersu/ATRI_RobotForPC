"""Adapt GPT-SoVITS api_v2 without changing the desktop audio protocol."""

import base64
import io
import json
import pickle
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from server import tts_config


# Fields supported by the installed api_v2 TTS_Request schema. Do not forward
# model_name or weight-switching fields: api_v2 uses its preloaded base models.
API_V2_FIELDS = frozenset({
    "text", "text_lang", "ref_audio_path", "prompt_lang", "prompt_text",
    "top_k", "top_p", "temperature", "text_split_method", "batch_size",
    "batch_threshold", "split_bucket", "speed_factor", "fragment_interval",
    "seed", "parallel_infer", "use_cuda_graph", "repetition_penalty",
    "sample_steps", "cfg_rate", "super_sampling", "overlap_length",
    "min_chunk_length",
})


def apply_pet_tts_config(pet_dir, requested_name, reference_path, payload):
    """Apply optional server-owned config/tts.json; never trust client paths."""
    config_dir = Path(pet_dir) / "config"
    profile_path = config_dir / "tts.json"
    outgoing = dict(payload)
    if not profile_path.is_file():
        return reference_path, outgoing
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read pet config/tts.json") from exc
    if not isinstance(profile, dict):
        raise ValueError("Pet TTS configuration must be an object")
    if profile.get("backend") and not outgoing.get("tts_backend"):
        outgoing["tts_backend"] = profile["backend"]
    if profile.get("prompt_lang"):
        outgoing["prompt_lang"] = profile["prompt_lang"]
    mode = profile.get("prompt_text_mode", "filename")
    if mode not in ("filename", "transcript"):
        raise ValueError("Unsupported pet prompt_text_mode")
    requested_stem = Path(str(requested_name)).stem
    if mode == "transcript":
        prompts = profile.get("reference_prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError("Pet reference_prompts must be an object")
        prompt = prompts.get(requested_stem)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing reference transcript for: {requested_stem}")
        outgoing["prompt_text"] = prompt.strip()
    if profile.get("reference_dir"):
        relative = profile["reference_dir"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Pet reference_dir must be relative to config")
        directory = (config_dir / relative).resolve()
        if not directory.is_relative_to(config_dir.resolve()):
            raise ValueError("Pet reference_dir cannot leave config")
        name = Path(str(requested_name)).stem
        aliases = profile.get("reference_aliases", {})
        if not isinstance(aliases, dict):
            raise ValueError("Pet reference_aliases must be an object")
        name = aliases.get(name, name)
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or chr(92) in name:
            raise ValueError("Invalid pet reference alias")
        reference_path = directory / (name + ".wav")
        if not reference_path.resolve().is_relative_to(directory):
            raise ValueError("Pet reference alias cannot leave reference_dir")
        if not reference_path.is_file():
            raise ValueError(f"Prepared reference audio not found: {reference_path}")
        if mode == "filename":
            prompt = reference_path.stem.partition("_")[2]
            if prompt:
                outgoing["prompt_text"] = prompt
    return reference_path, outgoing


def prepare_tts_request(payload):
    """Resolve a server-approved endpoint; never accept a client-supplied URL."""
    outgoing = dict(payload)
    backend = str(outgoing.pop("tts_backend", None) or tts_config.TTS_BACKEND).strip().lower()
    if backend == "new_api":
        return backend, tts_config.NEW_API_URL, outgoing, tts_config.NEW_API_TIMEOUT_SECONDS
    if backend != "api_v2":
        raise ValueError("Unsupported tts_backend; choose new_api or api_v2")

    outgoing = {key: value for key, value in outgoing.items() if key in API_V2_FIELDS}
    for field in ("text", "text_lang", "prompt_lang", "ref_audio_path"):
        if not isinstance(outgoing.get(field), str) or not outgoing[field].strip():
            raise ValueError(f"api_v2 requires {field}")
    # Reference transcription comes from the filename or the pet profile.
    # An explicit prompt remains valid for an unlabelled file.
    prompt = outgoing.get("prompt_text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "api_v2 requires reference prompt_text; name the file mood_transcript.wav "
            "or supply prompt_text explicitly"
        )
    for field in ("text_lang", "prompt_lang"):
        language = outgoing[field].strip().lower()
        outgoing[field] = {"jp": "ja", "cn": "zh"}.get(language, language)
    # Socket acknowledgements expect a complete audio object, not WAV chunks.
    outgoing["media_type"] = "wav"
    outgoing["streaming_mode"] = False
    return backend, tts_config.API_V2_URL, outgoing, tts_config.API_V2_TIMEOUT_SECONDS


def format_api_v2_response(content, return_format=""):
    """Return WAV/base64 for mobile, or the existing pickle contract for PC."""
    if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise ValueError("api_v2 did not return WAV audio; check its URL and service log")
    try:
        audio = AudioSegment.from_file(io.BytesIO(content), format="wav")
    except Exception as exc:
        raise ValueError("api_v2 returned invalid or incomplete WAV audio") from exc
    if not audio.raw_data:
        raise ValueError("api_v2 returned empty WAV audio")
    if return_format == "wav":
        return content
    if return_format == "base64":
        return {
            "ok": True, "format": "wav",
            "audio_base64": base64.b64encode(content).decode("ascii"),
        }
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    # voice_producer.get_radio reads data[-1][0] and data[-1][1].
    return pickle.dumps([(audio.frame_rate, samples)], protocol=4)
