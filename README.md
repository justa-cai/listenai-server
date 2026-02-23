# ListenAI Server

适合arcs_mini开发板的 多模块 AI 语音服务系统 - 支持 ASR(语音识别)、TTS(语音合成)、VAD(语音活动检测) 等功能

## 模块说明

| 模块 | 功能 | 主要技术 | 端口 |
|------|------|----------|------|
| `asr/` | 实时语音识别服务，支持流式音频输入 | FunASR-Nano, WebSocket, TenVAD | WS: 9200, HTTP: 9201 |
| `tts/` | 文本转语音，支持声音克隆和流式输出 | VoxCPM-0.5B, WebSocket, 模型推理池 | WS: 9300, WebUI: 9301 |
| `vad/` | 语音活动检测，识别语音段起止 | TEN Framework, ONNX | - |
| `llm/` | 大语言模型客户端，支持对话和工具调用 | OpenAI API, Token性能分析 | - |
| `music/` | 音乐HTTP服务，AI智能搜索，图片生成 | OpenAI API, SiliconFlow, Z-Image-Turbo | HTTP: 9100 |
| `cloud/` | LLM网关，会话管理，MCP工具调用 | WebSocket, MCP协议, aiohttp | WS: 9400 |


## 环境初始化

### 环境要求

- Python 3.10.x
- Linux (推荐 Ubuntu 20.04+)
- NVIDIA GPU + CUDA 12.x (可选，用于加速)

### 安装python依赖

```bash
# 1. 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境
uv venv --python 3.10
source .venv/bin/activate

# 3. 安装依赖
uv pip install -r requirements.txt

```

### 

## 运行服务

### 初始化环境
```bash
source .venv/bin/activate
```

### ASR
```bash
cd asr
sh auto.sh
```

### TTS
```
cd tts
sh auto.sh
```

### CLOUD
```bash
cd cloud
sh auto.sh
```

### MUSIC
```bash
cd music
sh auto.sh
```

---

<div align="center">

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=justa-cai/listenai-server&type=Date)](https://star-history.com/#justa-cai/listenai-server&Date)

</div>
