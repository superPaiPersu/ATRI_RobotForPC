# TTS 后端切换

本功能接在主服务 server/run.py -> server/socket_handlers.py 的 voice_produce 事件。
不修改客户端、pet_template/main.py，也不修改 non_streaming 备用项目。

## 服务端统一选择（客户端不需要调整）

修改 server/tts_config.py：

    TTS_BACKEND = "api_v2"
    API_V2_URL = "http://127.0.0.1:9881/tts"

然后重启桌宠服务端。以 tts_config.py 的实际值为准；当前本机已选择 api_v2，改回 new_api 即可回退。
new_api 地址仍为 http://127.0.0.1:9870/tts。两个服务使用不同端口，可以并存。
不要把 api_v2 写成 /api_v2 路由，它的合成路由同样是 /tts。

## 单次请求选择（可选）

在传给 ai_tool.voice_produce(dt) 的 dt 中增加：

    dt["tts_backend"] = "api_v2"

或选择 new_api；未传此字段则使用服务端 TTS_BACKEND。字段放在 Socket 请求的 data 对象内。
new_api 保留原 model_name；api_v2 不发送 model_name、不调用切换权重接口。

## 启动免训练参考音频服务

已核对本机安装目录及进程：E:/v5/GPT-SoVITS-v2pro-20261006，端口 9881。
需要重新启动时，在该目录用 PowerShell 执行：

    .\runtime\python.exe -I api_v2.py -a 127.0.0.1 -p 9881 -c GPT_SoVITS/configs/tts_infer.yaml

若 9881 已有 api_v2 服务，不要重复启动。桌宠服务端和 TTS 服务是两个进程。
请确认 tts_infer.yaml 中 custom 指向预训练基础权重，而不是角色训练权重。
本机检查时为 s1v3.ckpt + v2Pro/s2Gv2ProPlus.pth，version 为 v2ProPlus。
本次没有修改这些权重或 TTS 服务配置，也没有启动训练。

## 参考音频和返回格式

- 音频存放在 server_pets/<pet_id>/config/voice_refs/。
- 推荐命名为 normal_参考音频实际说出的文字.wav。不是待合成文本，转录必须准确。
- 客户端继续传 normal.wav；服务端优先精确匹配，不存在时匹配同情绪、同扩展名的带文本文件。
- prompt_text 从实际文件名第一个下划线后提取，保留其后的下划线和标点。
- 若使用没有下划线的文件名，需在请求中显式传 prompt_text。
- text_lang 是要生成的语言，prompt_lang 是参考录音的语言；两者不一定相同。
- 路径必须能被 api_v2 进程读取。若两个服务不在同一机器，需要共享文件和一致路径；本次不做跨机器上传。
- api_v2 固定请求非流式 WAV。每个文本分句仍可单独合成；本选项不改为音频分块流式播放。
- PC 默认响应转换回现有 pickle 的 [(sample_rate, samples)]，不改播放代码。
- return_format=wav/base64 保持移动端原协议；错误统一返回 ok=false/error/detail。

## 检查

    python -m unittest server.test_tts_prompt server.test_tts_backend server.test_pet_tts_config -v

单元测试不加载模型、不调用真实 TTS。真实音色、发音和性能请另做试听，不能由协议测试保证。

## 原文件名包含情绪说明的宠物（dokara）

- 保留 config/voice_refs 中原 OGG 文件及名称，不改客户端 mood 列表。
- config/tts_refs 保存 32kHz 单声道 PCM WAV 副本；短录音补尾部静音，长录音优先在停顿处截取到 3–10 秒范围。
- 用 GPT-SoVITS 自带运行环境的 Fun-ASR-Nano 识别日语；裁短的副本单独识别，不使用原录音全文冒充。
- config/voice_refs.list 是原始录音标注；config/tts_refs.list 是实际合成参考标注，格式均为 音频路径|dokara|JA|原文。
- config/tts.json 中 prompt_text_mode=transcript，reference_aliases 映射原 mood 到 WAV 文件，reference_prompts 存对应录音的真实转录。
- 显式转录优先于文件名解析；缺失转录时直接报错，不把中文情绪说明当日语、不发送空 prompt。
- 自动识别结果未逐条人工听校，可修改 reference_prompts 校正，须确保文本与对应 WAV 一致。
- tts_reference_manifest.json 记录源文件哈希、处理方式、两套转录及人工复核状态。
- 启用新的服务端适配需重启桌宠服务端；不需要改 pet_template/main.py，不需要重启已运行的 api_v2。
