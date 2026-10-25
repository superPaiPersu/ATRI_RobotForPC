import wave
import struct
import math
from pydub import AudioSegment
import numpy as np
def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:

        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:

        data = data / 65536
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:

        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:

        data = data * 257 - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.int8:

        data = data * 256
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data


def audio_to_file(sample_rate, data, filename, format="wav"):
    if format == "wav":
        data = convert_to_16_bit_wav(data)
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=(1 if len(data.shape) == 1 else data.shape[1]),
    )
    file = audio.export(filename, format=format)
    file.close()  # type: ignore

def analyze_wav(file_path, window_ms=100):
    """
    分析 WAV 音频文件，返回音量时间轴。

    参数：
        file_path : WAV 文件路径
        window_ms : 分析窗口长度（毫秒），默认 20ms

    返回：
        (volume_list, window_ms)
        volume_list : 每个窗口的响度值 (0.0 ~ 1.0)
        window_ms   : 窗口长度（毫秒），方便后续时间计算
    """
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()  # 采样率，如 44100
        n_channels = wf.getnchannels()  # 声道数
        sample_width = wf.getsampwidth()  # 样本位深（字节数）
        n_frames = wf.getnframes()  # 总帧数

        # 读取全部音频数据
        raw_data = wf.readframes(n_frames)

    # 将原始字节数据转换为整数样本列表
    if sample_width == 1:
        fmt = f"{n_frames * n_channels}B"  # 无符号 8 位
        samples = struct.unpack(fmt, raw_data)
        # 转换为有符号：减去 128
        samples = [s - 128 for s in samples]
    elif sample_width == 2:
        fmt = f"{n_frames * n_channels}h"  # 有符号 16 位
        samples = struct.unpack(fmt, raw_data)
    else:
        raise ValueError("只支持 8 位或 16 位 WAV 文件")

    # 将立体声转为单声道（取平均值），如果是单声道则直接使用
    if n_channels == 2:
        # 交错排列：L0,R0,L1,R1,...
        mono_samples = [(samples[i] + samples[i + 1]) / 2.0
                        for i in range(0, len(samples), 2)]
    else:
        mono_samples = list(samples)

    # 计算每个窗口的 RMS 值
    window_size = int(sample_rate * window_ms / 1000.0)  # 每个窗口包含的样本数
    volume_list = []

    for start in range(0, len(mono_samples), window_size):
        chunk = mono_samples[start:start + window_size]
        if not chunk:
            break
        # 计算 RMS：平方和除以长度再开方
        sum_sq = sum(x * x for x in chunk)
        rms = math.sqrt(sum_sq / len(chunk))
        volume_list.append(rms)

    # 归一化到 0~1 区间（基于该音频的最大 RMS）
    max_rms = max(volume_list) if volume_list else 1
    print(max_rms)
    if max_rms > 0:
        volume_list = [v / max_rms for v in volume_list]
    else:
        volume_list = [0.0] * len(volume_list)

    return (volume_list,window_ms)
