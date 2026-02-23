# ListenAI Server

多模块 AI 语音服务系统 - 支持 ASR(语音识别)、TTS(语音合成)、VAD(语音活动检测) 等功能

## 环境要求

- Python 3.10.x
- Linux (推荐 Ubuntu 20.04+)
- NVIDIA GPU + CUDA 12.x (可选，用于加速)

## 快速开始

```bash
# 1. 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境
uv venv --python 3.10
source .venv/bin/activate

# 3. 安装依赖
uv pip install -r requirements.txt

# 4. 安装本地模块 (可选)
uv pip install -e ./vad/ten-vad
```

## 模块说明

| 模块 | 功能 | 主要技术 |
|------|------|----------|
| `asr/` | 语音识别 | FunASR, Whisper, WeNet |
| `tts/` | 语音合成 | VoxCPM, pyopenjtalk |
| `vad/` | 语音活动检测 | TenVAD, RNNoise |
| `llm/` | 大语言模型 | OpenAI API |
| `music/` | 音乐处理 | Diffusers, Transformers |
| `cloud/` | 云服务 | |

## 技术栈

- **深度学习**: PyTorch 2.10, Transformers 5.2
- **Web 框架**: FastAPI, WebSockets
- **音频处理**: Librosa, SoundFile, FunASR
- **GPU 加速**: CUDA 12.x, cuDNN

## 详细文档

- [安装指南](INSTALL.md)
- ASR API: `asr/ASR_WEBSOCKET_API.md`
- TTS Voices: `tts/VOICES.md`

## License

Proprietary - All rights reserved
