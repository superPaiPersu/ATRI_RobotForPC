import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from server.qwen3_omni_local import Qwen3OmniLoader, Qwen3OmniLocalSession


SOURCE_AUDIO = Path(
    "/root/ai-pet-platform/server_pets/atri/config/prefab_audio/"
    "夏生先生明明那么厉害，有时候却很爱撒娇呢，但是，这样的夏生先生也很可爱呢.wav"
)
TEST_AUDIO = Path("/tmp/qwen3_omni_smoke_16k.wav")


def gpu_snapshot(label):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"GPU {label}: {result.stdout.strip()}", flush=True)


def main():
    print("MODEL_PATH", os.environ.get("QWEN3_MODEL_PATH"), flush=True)
    gpu_snapshot("before_load")
    load_started = time.monotonic()
    model, _ = Qwen3OmniLoader.load()
    print(f"LOAD_SECONDS {time.monotonic() - load_started:.3f}", flush=True)
    print("MODEL_DEVICE", model.device, flush=True)
    print("MODEL_DTYPE", model.dtype, flush=True)
    print(
        "DEVICE_MAP",
        json.dumps(getattr(model, "hf_device_map", {}), ensure_ascii=False),
        flush=True,
    )
    module_types = {}
    for module in model.modules():
        name = type(module).__name__
        if "Quant" in name or "Compressed" in name:
            module_types[name] = module_types.get(name, 0) + 1
    print("QUANT_MODULES", json.dumps(module_types, sort_keys=True), flush=True)
    gpu_snapshot("after_load")

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(SOURCE_AUDIO),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(TEST_AUDIO),
        ],
        check=True,
    )
    audio, sample_rate = sf.read(TEST_AUDIO, dtype="float32", always_2d=False)
    assert sample_rate == 16000
    print(
        "AUDIO",
        json.dumps(
            {
                "samples": int(audio.size),
                "seconds": float(audio.size / sample_rate),
                "rms": float(np.sqrt(np.mean(audio * audio))),
                "peak": float(np.max(np.abs(audio))),
            }
        ),
        flush=True,
    )

    chunks = []
    events = []
    generation_started = time.monotonic()
    first_chunk_at = [None]

    def on_chunk(flag, payload):
        now = time.monotonic()
        if first_chunk_at[0] is None:
            first_chunk_at[0] = now
        record = {"flag": flag, "payload": payload, "at": now - generation_started}
        chunks.append(record)
        print("CHUNK", json.dumps(record, ensure_ascii=False), flush=True)

    def on_event(payload):
        record = {"payload": payload, "at": time.monotonic() - generation_started}
        events.append(record)
        print("EVENT", json.dumps(record, ensure_ascii=False), flush=True)

    rules = (
        "你是亚托莉。用户会直接用语音和你对话。"
        "只输出连续的完整 JSON 对象，不要 Markdown，不要解释。"
        "先输出一个参数对象："
        '{"type":"params","action":"reply","mood":"normal",'
        '"expression":"normal","heart-beat-seconds":"30","special":"None"}。'
        "之后按自然短句逐个输出 segment 对象，每个对象必须同时包含日文回复 jp 和对应中文翻译 cn，seq 从 1 递增。"
        '最后输出且只输出 {"type":"end"}。'
    )
    session = Qwen3OmniLocalSession(
        messages=[
            {"role": "developer", "content": rules},
            {"role": "user", "content": "你好，你叫什么名字？"},
            {
                "role": "assistant",
                "content": (
                    '{"type":"params","action":"reply","mood":"normal",'
                    '"expression":"normal","heart-beat-seconds":"30","special":"None"}'
                    '{"type":"segment","seq":1,"jp":"私はアトリです。",'
                    '"cn":"我是亚托莉。"}{"type":"end"}'
                ),
            },
            {"role": "developer", "content": "再次严格遵守以上 JSON 输出规则。"},
        ],
        on_chunk=on_chunk,
        on_event=on_event,
    )
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    session._generate(pcm16)
    elapsed = time.monotonic() - generation_started
    print(f"GENERATE_SECONDS {elapsed:.3f}", flush=True)
    print(
        "FIRST_CHUNK_SECONDS",
        "none" if first_chunk_at[0] is None else f"{first_chunk_at[0] - generation_started:.3f}",
        flush=True,
    )
    print("CHUNK_FLAGS", json.dumps([item["flag"] for item in chunks]), flush=True)
    gpu_snapshot("after_generate")
    valid = (
        bool(chunks)
        and chunks[0]["flag"] == "params"
        and any(item["flag"] == "segment" for item in chunks)
        and chunks[-1]["flag"] == "end"
    )
    print("PROTOCOL_VALID", valid, flush=True)
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
