# ATRI_RobotForPC

Python AI 桌宠平台源码，包含 PC 客户端、Socket.IO 服务端、宠物配置与资源，以及 non_streaming 备用实现。

## 目录

- `client/`：桌宠管理界面、模板、文件与屏幕工具、Harness 桥接。
- `server/`：模型调用、实时语音、历史与标签、工具调度、TTS 适配。
- `server_pets/`：宠物规则、人设、Live2D 与参考音频资源。
- `non_streaming/`：备用非流式实现。
- `realtime_example/`：独立实时语音示例。

## 密钥配置

仓库不包含 API Key、登录凭据、Cookie 或私人聊天历史。发布版已将原硬编码密钥替换为环境变量读取；请求中传入 api_url/api_key 的既有接口保留。

`config.env.example` 只列出变量名，**不会自动加载**。例如在 PowerShell 中：

```powershell
$env:GEMINI_API_KEY = "填入自己的密钥"
$env:DEEPSEEK_API_KEY = "填入自己的密钥"
$env:OPENAI_API_KEY = "填入自己的密钥"
python server/run.py
```

按需设置 DEVIN_API_KEY、BOCHA_API_KEY、QWEN_REALTIME_API_KEY、GEMINI_REALTIME_API_KEY 等变量。多 Gemini Live 密钥可用 GEMINI_REALTIME_API_KEYS，以英文逗号分隔。客户端模型菜单使用对应环境变量，启动客户端前也需要配置。FLASK_SECRET_KEY 建议部署时设为固定随机值；不配置时进程生成临时值。不要把真实值提交进脚本。

## 本地运行

1. 按需安装 `requirements.txt` / `requirements-server.txt`；语音识别另见 `server/requirements-stt.txt`。
2. 配置模型地址、模型名与密钥。现有默认模型名和中转地址保留，请确认所用提供商实际支持。
3. 服务端可用 `python server/run.py` 或 `run_server.bat` 启动；PC 管理界面可用 `python client/main.py` 启动。客户端连接地址在 `client/config.json`，发布默认值为 `http://127.0.0.1:5000`。
4. TTS 配置参阅 `server/TTS_BACKENDS.md`；部署参阅 `SERVER_DEPLOY_WINDOWS.md`。部分本机模型/音频路径需要按实际环境调整。

这是源码快照，不是开箱即用的安装包。未上传打包 EXE、_internal 运行库、宠物 ZIP、模型权重及本地缓存；涉及下载/启动已打包桌宠的流程需要自行准备对应构建与宠物包。原始参考音频及宠物素材保留。

## 发布安全

推送前运行 `python scripts/check_public_secrets.py`。首次公开提交前已检查暂存内容并移除硬编码密钥、音乐 Cookie 示例，排除账号数据、聊天/摘要记录、日志、备份、模型权重及临时录音。自动扫描无法保证识别所有未知格式的秘密，发布前仍需人工检查。

第三方模型、角色、音乐及字体素材的权利归原权利人；本仓库不额外授予这些素材的使用授权。

