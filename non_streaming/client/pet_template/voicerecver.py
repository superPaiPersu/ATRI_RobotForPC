import os
import sounddevice as sd
import numpy as np
import asyncio
import queue

from google import genai
from google.genai import types

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

audio_queue = queue.Queue()

# ---------------- 接收 ----------------
async def receive_loop(session):
    async for message in session.receive():

        if message.server_content:
            content = message.server_content

            if content.input_transcription and content.input_transcription.text:
                print("原文:", content.input_transcription.text)

        # if content.output_transcription:
        #     print("翻译:", content.output_transcription.text)

        # if content.model_turn:
        #     for part in content.model_turn.parts:
        #         if part.inline_data:
        #             audio = part.inline_data.data
        #             print("收到翻译音频:", len(audio))


# ---------------- 发送 ----------------
async def send_audio(session, loop):

    def callback(indata, frames, time, status):
        if status:
            print(status)

        pcm = (indata * 32767).astype(np.int16).tobytes()
        audio_queue.put(pcm)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=callback
    ):
        print("开始说话...")
        while True:
            pcm = await loop.run_in_executor(None, audio_queue.get)

            await session.send_realtime_input(
                audio=types.Blob(
                    data=pcm,
                    mime_type="audio/pcm;rate=16000"
                )
            )


# ---------------- 主函数 ----------------
async def main():
    client = genai.Client(api_key=os.getenv("GEMINI_REALTIME_API_KEY", "").strip())

    MODEL = "gemini-3.5-live-translate-preview"

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),

    )

    loop = asyncio.get_running_loop()

    async with client.aio.live.connect(
        model=MODEL,
        config=config
    ) as session:

        recv_task = asyncio.create_task(receive_loop(session))
        send_task = asyncio.create_task(send_audio(session, loop))

        await asyncio.gather(recv_task, send_task)


# ---------------- 启动 ----------------
if __name__ == "__main__":
    asyncio.run(main())