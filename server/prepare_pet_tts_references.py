"""Generate TTS-only WAV copies; never overwrite original voice_refs."""

import argparse
import hashlib
import json
from pathlib import Path

from pydub import AudioSegment, silence


MIN_REFERENCE_MS = 3200
MAX_REFERENCE_MS = 9500


def prepare_references(pet_dir):
    pet_dir = Path(pet_dir)
    source_dir = pet_dir / "config" / "voice_refs"
    output_dir = pet_dir / "config" / "tts_refs"
    records = {}
    sources = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in (".ogg", ".wav", ".mp3", ".flac"))
    if not sources:
        raise ValueError("No reference audio files found")
    stems = [p.stem for p in sources]
    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate reference stems would overwrite output WAV files")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        audio = AudioSegment.from_file(str(source)).set_channels(1).set_frame_rate(32000).set_sample_width(2)
        original_ms = len(audio)
        if not original_ms or audio.rms == 0:
            raise ValueError(f"Empty or silent reference: {source.name}")
        change = "converted"
        if len(audio) > MAX_REFERENCE_MS:
            pauses = silence.detect_silence(
                audio[:MAX_REFERENCE_MS], min_silence_len=150,
                silence_thresh=max(audio.dBFS - 16, -50), seek_step=10,
            )
            ends = [(start + end) // 2 for start, end in pauses
                    if MIN_REFERENCE_MS <= (start + end) // 2 <= MAX_REFERENCE_MS]
            audio = audio[:max(ends) if ends else MAX_REFERENCE_MS]
            change = "trimmed_at_pause" if ends else "trimmed_at_limit"
        elif len(audio) < MIN_REFERENCE_MS:
            audio += AudioSegment.silent(duration=MIN_REFERENCE_MS-len(audio), frame_rate=audio.frame_rate)
            change = "silence_padded"
        output = output_dir / (source.stem + ".wav")
        handle = audio.export(str(output), format="wav")
        handle.close()
        records[source.name] = {
            "file": output.name, "original_ms": original_ms, "prepared_ms": len(audio),
            "operation": change, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    (pet_dir / "config" / "tts_reference_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pet_dir", type=Path)
    args = parser.parse_args()
    records = prepare_references(args.pet_dir)
    print(json.dumps({"files": len(records), "pet_dir": str(args.pet_dir)}, ensure_ascii=False))
