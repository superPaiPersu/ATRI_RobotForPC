# GPT 实时语音对话示例

这是一个独立的小示例：麦克风音频实时发送给 OpenAI Realtime API，模型的语音分片实时播放。它不依赖桌宠项目的 Socket.IO、服务端路由或 Harness 链路。

## 安装

在项目根目录执行：

```powershell
python -m pip install -r realtime_example\requirements.txt
```

## 设置密钥并运行

PowerShell 当前窗口：

```powershell
$env:OPENAI_API_KEY = "你的 OpenAI API Key"
python realtime_example\realtime_voice_chat.py
```

程序会自动使用默认麦克风和扬声器。说话后停顿约半秒，模型会自动开始回答；按 `Ctrl+C` 退出。

如果默认设备不正确，可先在 Python 中查看设备：

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

脚本当前使用 `gpt-realtime`、24 kHz 单声道 PCM16 和服务端 VAD。模型直接接收麦克风音频并生成语音，控制台会同时流式显示模型语音的文本版本。

为了避免扬声器声音再次进入麦克风，建议戴耳机测试。
