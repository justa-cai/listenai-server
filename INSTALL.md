# ListenAI Server 安装指南

## 系统要求

- **Python**: 3.10.x (推荐 3.10.12+)
- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **GPU**: NVIDIA GPU (可选，用于加速 TTS/ASR 推理)
  - CUDA 12.x 支持
  - 建议 8GB+ VRAM

## 快速安装

### 1. 安装 uv (推荐)

```bash
# 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
uv venv --python 3.10

# 激活虚拟环境
source .venv/bin/activate

# 安装所有依赖
uv pip install -r requirements.txt
```

### 3. 安装本地模块 (可选)

如果需要 VAD (语音活动检测) 功能：

```bash
uv pip install -e ./vad/ten-vad
```

### 4. 验证安装

```bash
# 检查核心包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import funasr; print(f'FunASR: {funasr.__version__}')"
python -c "import websockets; print('WebSockets OK')"
```

## 各模块安装

### ASR (语音识别)
```bash
cd asr
# 参考 asr/README.md
```

### TTS (语音合成)
```bash
cd tts
uv pip install -e .
```

### Cloud (云服务)
```bash
cd cloud
cp .env.example .env
# 配置环境变量
```

## 常见问题

### CUDA/GPU 支持
如果需要使用 GPU 加速，确保已安装：
- NVIDIA Driver (535.x+)
- CUDA Toolkit 12.x

安装后可验证：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 网络问题
如果下载缓慢，可使用国内镜像：
```bash
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 目录结构

```
listenai_server/
├── asr/          # 语音识别服务
├── tts/          # 语音合成服务
├── vad/          # 语音活动检测
├── llm/          # 大语言模型
├── music/        # 音乐/歌词处理
├── cloud/        # 云服务
└── arcs_mini/    # 嵌入式 SDK
```

## 卸载

```bash
# 删除虚拟环境
rm -rf .venv
```
