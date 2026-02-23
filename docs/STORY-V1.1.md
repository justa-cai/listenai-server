# ListenAI Cloud Agent V1.1 - 解耦架构需求文档

## 版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| V1.0 | 2026-02-19 | 初始版本，耦合架构 |
| V1.1 | 2026-02-20 | 解耦架构，服务独立 |

## 概述

ListenAI Voice Assistant V1.1 采用完全解耦的架构设计，将 ASR、LLM、TTS 三个核心服务完全独立，客户端直接与各服务通信，Cloud Agent 仅作为 LLM 服务网关。

## 架构变更

### V1.0 架构（耦合）

```
客户端 ──WebSocket──► Cloud Agent (:9400)
                           ├──► ASR Service (:9200)
                           ├──► LLM Service (:8000)
                           └──► TTS Service (:9300)
```

### V1.1 架构（解耦）

```
客户端 ──WebSocket──► ASR Service (:9200)    [语音识别]
客户端 ──WebSocket──► Cloud Agent (:9400)    [LLM 网关]
                           └──► LLM Service (:8000)
客户端 ──WebSocket──► TTS Service (:9300)    [语音合成]
```

## 服务职责

### 1. ASR Service (:9200) - 语音识别服务

**职责**：接收客户端音频，返回识别文本

**通信协议**：
- WebSocket 服务
- 接收：二进制 PCM 音频数据（16-bit, 16kHz, mono）
- 返回：JSON 格式识别结果

**客户端直接访问**，无需经过 Cloud Agent。

### 2. Cloud Agent (:9400) - LLM 服务网关

**职责**：
- LLM 对话管理
- 会话上下文管理
- MCP 工具调用
- HTTP 健康检查

**通信协议**：
- WebSocket 服务（纯文本通信）
- HTTP 服务（健康检查）

**消息类型**：

| 类型 | 方向 | 说明 |
|------|------|------|
| `text_input` | 客户端→服务端 | 用户输入文本 |
| `llm_response` | 服务端→客户端 | LLM 响应内容 |
| `tool_call` | 服务端→客户端 | 工具调用结果 |
| `start_session` | 客户端→服务端 | 开始会话 |
| `end_session` | 客户端→服务端 | 结束会话 |
| `configure` | 客户端→服务端 | 配置参数 |
| `status` | 服务端→客户端 | 状态更新 |
| `error` | 服务端→客户端 | 错误信息 |
| `ping/pong` | 双向 | 心跳 |

### 3. TTS Service (:9300) - 语音合成服务

**职责**：接收客户端文本，返回合成音频

**通信协议**：
- WebSocket 服务 (`ws://host:9300/tts`)
- 接收：JSON 格式 TTS 请求
- 返回：二进制音频流（VoxCPM 帧格式）

**客户端直接访问**，无需经过 Cloud Agent。

## 数据流

### 语音识别流程

```
1. 客户端建立录音
2. 客户端连接 ASR Service (:9200)
3. 客户端发送音频数据到 ASR
4. ASR 返回识别结果（JSON）
5. 客户端显示识别文本
6. 识别完成时，进入 LLM 流程
```

### LLM 对话流程

```
1. 客户端连接 Cloud Agent (:9400)
2. 客户端发送 text_input（包含 ASR 识别文本）
3. Cloud Agent 调用 LLM Service
4. Cloud Agent 返回 llm_response（可能包含 tool_call）
5. 客户端显示 LLM 响应
6. 如需播放，进入 TTS 流程
```

### 语音合成流程

```
1. 客户端连接 TTS Service (:9300)
2. 客户端发送 tts_request（包含 LLM 响应文本）
3. TTS 返回音频流（二进制帧）
4. 客户端解析并播放音频
```

## 协议定义

### Cloud Agent WebSocket 协议

#### 客户端消息

**text_input - 文本输入**
```json
{
  "type": "text_input",
  "text": "今天天气怎么样",
  "session_id": "optional-session-id"
}
```

**start_session - 开始会话**
```json
{
  "type": "start_session",
  "session_id": "optional-existing-session-id"
}
```

**end_session - 结束会话**
```json
{
  "type": "end_session"
}
```

**configure - 配置**
```json
{
  "type": "configure",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**ping - 心跳**
```json
{
  "type": "ping"
}
```

#### 服务端消息

**llm_response - LLM 响应**
```json
{
  "type": "llm_response",
  "content": "今天北京天气晴朗...",
  "is_final": true,
  "timestamp": "2026-02-20T10:00:00Z"
}
```

**tool_call - 工具调用**
```json
{
  "type": "tool_call",
  "tool_name": "get_weather",
  "arguments": {"city": "Beijing"},
  "result": {"temperature": 15, "condition": "Sunny"},
  "success": true,
  "duration_ms": 150.5,
  "timestamp": "2026-02-20T10:00:00Z"
}
```

**status - 状态更新**
```json
{
  "type": "status",
  "status": "processing",
  "data": {},
  "timestamp": "2026-02-20T10:00:00Z"
}
```

**error - 错误**
```json
{
  "type": "error",
  "code": "LLM_ERROR",
  "message": "Failed to process request",
  "details": "Connection timeout",
  "timestamp": "2026-02-20T10:00:00Z"
}
```

**pong - 心跳响应**
```json
{
  "type": "pong",
  "timestamp": "2026-02-20T10:00:00Z"
}
```

## 配置

### 环境变量

**Cloud Agent 服务端配置**
```
CLOUD_HOST=0.0.0.0
CLOUD_PORT=9400
CLOUD_PING_INTERVAL=30
CLOUD_PING_TIMEOUT=300
CLOUD_MAX_CONNECTIONS=100
CLOUD_SESSION_TIMEOUT=3600
CLOUD_LOG_LEVEL=INFO
```

**LLM 服务配置**
```
LLM_BASE_URL=http://192.168.13.228:8000/v1/
LLM_MODEL=Qwen3-30B-A3B
LLM_API_KEY=123
LLM_TIMEOUT=120
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
```

**MCP 配置**
```
MCP_ENABLED=true
MCP_SERVER_NAME=arcs-mini-mcp-server
MCP_SERVER_VERSION=1.0.0
MCP_PROTOCOL_VERSION=2024-11-05
```

### 客户端配置

```
ASR_SERVICE_URL=ws://192.168.1.169:9200
CLOUD_AGENT_URL=ws://192.168.1.169:9400
TTS_SERVICE_URL=ws://192.168.1.169:9300/tts
```

## 文件结构

```
listenai_server/
├── cloud/
│   ├── .env                    # 配置文件
│   ├── requirements.txt        # Python 依赖
│   ├── test_client.html        # Web 测试客户端
│   └── src/
│       ├── __init__.py         # 包初始化
│       ├── __main__.py         # 启动入口
│       ├── config.py           # 配置管理（简化版）
│       ├── server.py           # WebSocket 服务器（简化版）
│       ├── llm_client.py       # LLM HTTP 客户端
│       ├── mcp_manager.py      # MCP 工具管理器
│       ├── session.py          # 会话管理
│       └── protocol.py         # 消息协议定义（简化版）
├── asr/                        # ASR 服务（独立）
├── tts/                        # TTS 服务（独立）
└── STORY-V1.1.md              # 本文档
```

## 客户端实现

### Web 测试客户端功能

1. **连接管理**
   - ASR 连接（独立）
   - Cloud Agent 连接（独立）
   - TTS 连接（独立）
   - 各服务连接状态独立显示

2. **音频录制**
   - 麦克风录音
   - 音频可视化
   - 直接发送到 ASR 服务

3. **ASR 结果处理**
   - 显示实时识别结果
   - 识别完成后自动触发 LLM

4. **LLM 对话**
   - 发送文本到 Cloud Agent
   - 显示响应内容
   - 显示工具调用结果
   - 可选触发 TTS 播放

5. **TTS 播放**
   - 发送文本到 TTS 服务
   - 流式播放音频
   - 音频播放状态显示

6. **会话管理**
   - 会话 ID 显示
   - 消息历史
   - 统计信息

## 优势

1. **服务解耦**：ASR、LLM、TTS 完全独立，可独立部署、扩展、维护
2. **灵活定制**：客户端可自由组合服务，实现不同业务流程
3. **职责清晰**：每个服务专注于自己的核心功能
4. **易于扩展**：可轻松替换任一服务的实现
5. **降低复杂度**：Cloud Agent 代码量大幅减少
6. **提高可用性**：任一服务故障不影响其他服务

## 变更日志

### V1.1 (2026-02-20)

**新增**
- Cloud Agent 纯文本 WebSocket 协议
- `text_input` 消息类型
- 独立的服务连接管理

**变更**
- Cloud Agent 移除 ASR/TTS 代理功能
- 客户端直接连接 ASR/TTS 服务
- 协议简化，移除音频相关消息类型

**删除**
- `asr_client.py` - 不再需要 ASR 客户端代理
- `tts_client.py` - 不再需要 TTS 客户端代理
- `audio_buffer.py` - 不再需要音频缓冲
- `AUDIO_DATA`、`ASR_RESULT`、`TTS_AUDIO` 消息类型
- `ASRConfig`、`TTSConfig` 配置类

---

**文档版本**: V1.1  
**最后更新**: 2026-02-20  
**状态**: 待实现
