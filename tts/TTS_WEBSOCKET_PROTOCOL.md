# VoxCPM WebSocket TTS Server Protocol

## 1. 概述

本文档描述了 VoxCPM WebSocket TTS 服务器的通信协议。该协议基于 WebSocket，支持流式和非流式两种文本转语音模式。

## 2. 连接

### 2.1 连接 URL

```
ws://<host>:<port>/tts
```

### 2.2 连接请求头

| Header | 说明 | 必需 |
|--------|------|------|
| `X-Client-ID` | 客户端标识 | 可选 |
| `X-Request-Timeout` | 请求超时时间（秒） | 可选 |

## 3. 消息格式

### 3.1 消息方向

```
Client ──────────────────────────────► Server
  │                                    │
  │  ┌─────────────────────────────┐  │
  │  │      TTS Request            │  │
  │  │      (JSON)                 │  │
  │  └─────────────────────────────┘  │
  │                                    │
  │  ┌─────────────────────────────┐  │
  │  │      Cancel Request         │  │
  │  │      (JSON)                 │  │
  │  └─────────────────────────────┘  │
  │                                    │
Server ──────────────────────────────► Client
  │                                    │
  │  ┌─────────────────────────────┐  │
  │  │      Audio Data             │  │
  │  │      (Binary)               │  │
  │  └─────────────────────────────┘  │
  │                                    │
  │  ┌─────────────────────────────┐  │
  │  │      Progress/Status        │  │
  │  │      (JSON)                 │  │
  │  └─────────────────────────────┘  │
  │                                    │
  │  ┌─────────────────────────────┐  │
  │  │      Error                  │  │
  │  │      (JSON)                 │  │
  │  └─────────────────────────────┘  │
```

### 3.2 消息类型枚举

```javascript
// 客户端 → 服务器
const ClientMessageType = {
  TTS_REQUEST: 'tts_request',
  CANCEL: 'cancel',
  PING: 'ping'
};

// 服务器 → 客户端
const ServerMessageType = {
  AUDIO_DATA: 'audio_data',
  PROGRESS: 'progress',
  COMPLETE: 'complete',
  ERROR: 'error',
  PONG: 'pong'
};
```

## 4. 客户端请求

### 4.1 TTS 请求

```json
{
  "type": "tts_request",
  "request_id": "uuid-string",
  "params": {
    "text": "要转换的文本内容",
    "mode": "streaming",
    "voice_id": "doubao-通用女声",
    "prompt_wav_path": null,
    "prompt_text": null,
    "cfg_value": 2.0,
    "inference_timesteps": 30,
    "normalize": false,
    "denoise": true,
    "retry_badcase": true,
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6.0
  }
}
```

#### 请求字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定值 `"tts_request"` |
| `request_id` | string | 是 | 请求唯一标识符（UUID） |
| `params` | object | 是 | TTS 参数对象 |

#### TTS 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 要转换为语音的文本内容（最大5000字符） |
| `mode` | string | 否 | `"streaming"` | 模式：`"streaming"` 或 `"non_streaming"` |
| `voice_id` | string | 否 | `null` | 声音ID，格式为 `"category-voice_name"` |
| `prompt_wav_path` | string \| null | 否 | `null` | 参考音频路径（直接指定，与voice_id二选一） |
| `prompt_text` | string \| null | 否 | `null` | 参考文本 |
| `cfg_value` | float | 否 | `2.0` | LM guidance 值（0.1-10.0），越高越符合参考音色 |
| `inference_timesteps` | int | 否 | `30` | 推理步数（1-50），越高越稳定但越慢 |
| `normalize` | boolean | 否 | `false` | 是否启用文本规范化 |
| `denoise` | boolean | 否 | `true` | 是否启用音频降噪 |
| `retry_badcase` | boolean | 否 | `true` | 是否启用坏案例重试 |
| `retry_badcase_max_times` | int | 否 | `3` | 最大重试次数（0-10） |
| `retry_badcase_ratio_threshold` | float | 否 | `6.0` | 坏案例检测的长度限制（1.0-20.0） |

#### 声音克隆 (voice_id)

使用 `voice_id` 参数进行声音克隆时，服务器会自动解析对应的音频文件和参考文本：

```json
{
  "type": "tts_request",
  "request_id": "uuid",
  "params": {
    "text": "你好，世界！",
    "voice_id": "doubao-通用女声"
  }
}
```

声音ID格式：`"{category}-{voice_name}"`

服务器会：
1. 从声音管理器查找对应的声音
2. 自动设置 `prompt_wav_path` 和 `prompt_text`
3. 使用该声音进行TTS生成

### 4.2 取消请求

```json
{
  "type": "cancel",
  "request_id": "uuid-string"
}
```

### 4.3 心跳请求

```json
{
  "type": "ping",
  "timestamp": 1234567890
}
```

## 5. 服务器响应

### 5.1 音频数据

**流式模式：** 分片发送音频数据

```
Frame Header (4 bytes): [0xAA, 0x55, 0x00, 0x01]
┌──────────────┬───────────────┬────────────────────┐
│ Magic (2B)   │ Msg Type (1B) │ Reserved (1B)      │
│ 0xAA 0x55    │ 0x01          │ 0x00               │
└──────────────┴───────────────┴────────────────────┘

Metadata (JSON, length-prefixed with 4 bytes)
{
  "request_id": "uuid-string",
  "sequence": 0,
  "sample_rate": 24000,
  "is_final": false
}

Audio Payload (Binary, length-prefixed with 4 bytes)
<raw audio bytes: PCM 16-bit, mono>
```

**非流式模式：** 一次性发送完整音频

```
Frame Header (4 bytes): [0xAA, 0x55, 0x00, 0x02]
┌──────────────┬───────────────┬────────────────────┐
│ Magic (2B)   │ Msg Type (1B) │ Reserved (1B)      │
│ 0xAA 0x55    │ 0x02          │ 0x00               │
└──────────────┴───────────────┴────────────────────┘

Metadata (JSON, length-prefixed with 4 bytes)
{
  "request_id": "uuid-string",
  "sample_rate": 24000,
  "duration": 5.23
}

Audio Payload (Binary, length-prefixed with 4 bytes)
<raw audio bytes: PCM 16-bit, mono>
```

#### 帧类型定义

| 值 | 类型 | 说明 |
|----|------|------|
| 0x01 | STREAMING_CHUNK | 流式音频分片 |
| 0x02 | NON_STREAMING | 非流式完整音频 |
| 0x03 | METADATA_ONLY | 元数据（用于开始通知） |

### 5.2 进度消息

```json
{
  "type": "progress",
  "request_id": "uuid-string",
  "state": "processing",
  "progress": 0.5,
  "message": "Generating audio..."
}
```

#### 状态定义

| 状态 | 说明 |
|------|------|
| `queued` | 请求已排队 |
| `processing` | 正在处理 |
| `generating` | 正在生成音频 |
| `encoding` | 正在编码 |
| `completed` | 生成完成 |
| `failed` | 生成失败 |
| `cancelled` | 已取消 |

### 5.3 完成消息

```json
{
  "type": "complete",
  "request_id": "uuid-string",
  "result": {
    "duration": 5.23,
    "sample_rate": 24000,
    "samples": 125520,
    "chunks": 10
  }
}
```

### 5.4 错误消息

```json
{
  "type": "error",
  "request_id": "uuid-string",
  "error": {
    "code": "INVALID_PARAMS",
    "message": "Invalid parameter: text is required",
    "details": {}
  }
}
```

#### 错误代码定义

| 错误代码 | HTTP 等效 | 说明 |
|----------|----------|------|
| `INVALID_PARAMS` | 400 | 参数无效 |
| `INVALID_JSON` | 400 | JSON 格式无效 |
| `TEXT_TOO_LONG` | 400 | 文本过长 |
| `UNSUPPORTED_FORMAT` | 400 | 不支持的格式 |
| `VOICE_NOT_FOUND` | 404 | 指定的声音不存在 |
| `VOICE_MANAGER_NOT_AVAILABLE` | 503 | 声音管理器未初始化 |
| `MODEL_NOT_LOADED` | 503 | 模型未加载 |
| `QUEUE_FULL` | 503 | 请求队列已满 |
| `GENERATION_FAILED` | 500 | 生成失败 |
| `TIMEOUT` | 504 | 请求超时 |
| `RATE_LIMITED` | 429 | 请求频率限制 |
| `UNKNOWN_MESSAGE_TYPE` | 400 | 未知的消息类型 |
| `INTERNAL_ERROR` | 500 | 内部错误 |

### 5.5 心跳响应

```json
{
  "type": "pong",
  "timestamp": 1234567890,
  "server_time": 1234567891
}
```

## 6. 消息流程示例

### 6.1 流式模式流程

```
Client                              Server
  │                                   │
  ├───── tts_request ─────────────────>│
  │                                   │
  │<──── progress (queued) ────────────┤
  │                                   │
  │<──── progress (generating) ────────┤
  │                                   │
  │<──── audio_chunk [seq=0] ──────────┤
  │<──── audio_chunk [seq=1] ──────────┤
  │<──── audio_chunk [seq=2] ──────────┤
  │                                   │
  │<──── complete ─────────────────────┤
  │                                   │
```

### 6.2 非流式模式流程

```
Client                              Server
  │                                   │
  ├───── tts_request ─────────────────>│
  │                                   │
  │<──── progress (processing) ────────┤
  │                                   │
  │<──── audio_full (complete) ────────┤
  │                                   │
  │<──── complete ─────────────────────┤
  │                                   │
```

### 6.3 取消流程

```
Client                              Server
  │                                   │
  ├───── tts_request ─────────────────>│
  │                                   │
  │<──── progress (processing) ────────┤
  │                                   │
  ├───── cancel ──────────────────────>│
  │                                   │
  │<──── complete (cancelled) ─────────┤
  │                                   │
```

## 7. 二进制帧格式详解

### 7.1 字节序

所有多字节整数使用 **大端序 (Big Endian)** 网络字节序。

### 7.2 帧结构

```
┌───────────────────────────────────────────────────────────┐
│                        Frame Header                        │
├──────────┬───────────┬────────────┬───────────────────────┤
│ Magic    │ Msg Type  │ Reserved   │ Metadata Length       │
│ (2 bytes)│ (1 byte)  │ (1 byte)   │ (4 bytes)             │
│ 0xAA 0x55│ 0x01-0x03 │ 0x00       │ N bytes               │
└──────────┴───────────┴────────────┴───────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                      Metadata (JSON)                       │
├───────────────────────────────────────────────────────────┤
│ UTF-8 encoded JSON string                                  │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                    Payload Length                          │
├───────────────────────────────────────────────────────────┤
│ 4 bytes, big endian                                       │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                      Audio Payload                         │
├───────────────────────────────────────────────────────────┤
│ Raw PCM audio data (16-bit, mono, native endianness)      │
└───────────────────────────────────────────────────────────┘
```

### 7.3 示例解析

```
Received: AA 55 01 00 00 00 3B 7B 22 72 65 71 ...

分解:
[AA 55]           - Magic Number
[01]              - Message Type (STREAMING_CHUNK)
[00]              - Reserved
[00 00 00 3B]     - Metadata Length (59 bytes)
[7B 22 72 65 71...] - Metadata JSON: {"request_id":"...",...}
[00 00 10 00]     - Payload Length (4096 bytes)
[<4096 bytes>]    - Audio Payload
```

## 8. HTTP API

服务器同时提供 HTTP 接口用于 Web UI 和声音管理，运行在 WebSocket 端口 +1（默认 9301）。

### 8.1 基础接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web UI 首页 |
| `/api/config` | GET | 获取客户端配置信息 |
| `/static/*` | GET | 静态资源文件 |

### 8.2 声音管理 API

#### 获取声音列表

```
GET /api/voices?category={category}&search={keyword}
```

**查询参数：**
- `category` (可选): 按分类筛选
- `search` (可选): 搜索关键词

**响应：**
```json
{
  "voices": {
    "doubao": [
      {
        "id": "doubao-通用女声",
        "name": "通用女声",
        "category": "doubao",
        "audio_path": "/path/to/voice_clone/doubao/通用女声.mp3",
        "sample_text": "参考文本内容"
      }
    ],
    "其他分类": [...]
  }
}
```

#### 获取声音分类

```
GET /api/voices/categories
```

**响应：**
```json
{
  "categories": ["doubao", "其他分类"]
}
```

#### 获取声音统计

```
GET /api/voices/stats
```

**响应：**
```json
{
  "total_voices": 10,
  "total_categories": 3,
  "voices_by_category": {
    "doubao": 5,
    "其他分类": 5
  }
}
```

#### 获取声音音频文件

```
GET /api/voices/{voice_id}/audio
```

返回指定声音的音频文件（用于试听）。

### 8.3 客户端配置

```
GET /api/config
```

**响应：**
```json
{
  "websocket_url": "ws://localhost:9300/tts",
  "default_params": {
    "mode": "streaming",
    "cfg_value": 2.0,
    "inference_timesteps": 30,
    "normalize": false,
    "denoise": true,
    "retry_badcase": true
  },
  "constraints": {
    "max_text_length": 5000,
    "cfg_value_range": [0.1, 10.0],
    "inference_timesteps_range": [1, 50]
  }
}
```

## 9. 配置参数

### 9.1 服务器默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `192.168.1.169` | 服务器绑定地址 |
| `port` | `9300` | WebSocket 端口 |
| `web_port` | `9301` | Web UI 端口 (port + 1) |
| `ping_interval` | `30` | WebSocket ping 间隔（秒） |
| `ping_timeout` | `300` | WebSocket ping 超时（5分钟） |
| `max_message_size` | `1 MB` | 单个消息大小限制 |
| `max_connections` | `100` | 最大并发连接数 |
| `max_concurrent_requests` | `10` | 最大并发请求数 |
| `max_queue_size` | `50` | 最大队列大小 |
| `request_timeout` | `600` | 请求超时（10分钟） |
| `chunk_size` | `4096` | 流式音频块大小（采样点数） |
| `sample_rate` | `24000` | 默认采样率 |
| `voice_dir` | `voice_clone` | 声音样本目录 |
| `debug_audio` | `false` | 调试音频保存 |
| `rate_limit_enabled` | `true` | 是否启用速率限制 |
| `rate_limit_per_minute` | `60` | 每分钟请求限制 |
| `metrics_enabled` | `false` | Prometheus 监控 |
| `metrics_port` | `9090` | 监控端口 |
| `log_level` | `INFO` | 日志级别 |
| `log_format` | `json` | 日志格式 |

### 9.2 模型默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `VoxCPM-0.5B` | 模型路径或 HuggingFace ID |
| `device` | `cuda` | 推理设备 |
| `cache_dir` | `null` | 模型缓存目录 |
| `trust_remote_code` | `true` | 信任远程代码 |
| `default_cfg_value` | `2.0` | 默认 CFG 值 |
| `default_inference_timesteps` | `30` | 默认推理步数 |
| `default_normalize` | `false` | 默认文本规范化 |
| `default_denoise` | `true` | 默认降噪 |
| `default_retry_badcase` | `true` | 默认坏案例重试 |
| `default_retry_badcase_max_times` | `3` | 最大重试次数 |
| `default_retry_badcase_ratio_threshold` | `6.0` | 坏案例检测阈值 |
| `max_text_length` | `5000` | 最大文本长度 |

### 9.3 环境变量

所有配置可通过环境变量设置，前缀为 `TTS_`：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `TTS_HOST` | 服务器地址 | `192.168.1.169` |
| `TTS_PORT` | WebSocket 端口 | `9300` |
| `TTS_MODEL_NAME` | 模型名称 | `VoxCPM-0.5B` |
| `TTS_DEVICE` | 推理设备 | `cuda` |
| `TTS_MAX_CONCURRENT` | 最大并发 | `10` |
| `TTS_MAX_CONNECTIONS` | 最大连接数 | `100` |
| `TTS_VOICE_DIR` | 声音目录 | `voice_clone` |
| `TTS_DEBUG_AUDIO` | 调试音频 | `false` |
| `TTS_LOG_LEVEL` | 日志级别 | `INFO` |
| `TTS_METRICS_ENABLED` | 监控开关 | `false` |

### 9.4 命令行参数

| 参数 | 说明 |
|------|------|
| `--debug-audio` | 启用调试音频保存 |

### 9.6 连接限制

| 限制项 | 值 | 说明 |
|--------|-----|------|
| 最大连接数 | 100 | 单个 WebSocket 服务器 |
| 消息大小限制 | 1 MB | 单个 WebSocket 消息 |
| 速率限制 | 10 req/min | 每个 IP 地址 |

## 10. 安全考虑

### 10.1 认证（可选）

如果启用认证，连接时需要提供 Token：

```
ws://<host>:<port>/tts?token=<jwt_token>
```

或通过子协议：

```
new WebSocket("ws://<host>:<port>/tts", ["voxcpm-tts", "Bearer.<jwt_token>"])
```

### 10.2 输入验证

- 文本长度限制
- 参数范围检查
- 防止注入攻击

### 10.3 速率限制

基于 IP 和 Token 的请求频率限制。

## 11. 客户端实现示例

### Python 客户端

```python
import asyncio
import websockets
import json
import uuid

async def voxcpm_tts(text, url="ws://localhost:9300/tts"):
    async with websockets.connect(url) as ws:
        request = {
            "type": "tts_request",
            "request_id": str(uuid.uuid4()),
            "params": {
                "text": text,
                "mode": "streaming"
            }
        }

        await ws.send(json.dumps(request))

        audio_chunks = []
        while True:
            message = await ws.recv()

            if isinstance(message, bytes):
                # Handle binary audio data
                chunks = parse_binary_frame(message)
                audio_chunks.extend(chunks)

                # Check if complete
                response = await ws.recv()
                response = json.loads(response)

                if response.get("type") == "complete":
                    break
            else:
                # Handle JSON control messages
                msg = json.loads(message)
                if msg.get("type") == "error":
                    raise Exception(msg["error"])

        return b"".join(audio_chunks)
```

### JavaScript 客户端

```javascript
class VoxCPMTTSClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
  }

  async speak(text, onAudioChunk, onProgress) {
    this.ws = new WebSocket(this.url);
    const requestId = crypto.randomUUID();

    return new Promise((resolve, reject) => {
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.ws.send(JSON.stringify({
          type: 'tts_request',
          request_id: requestId,
          params: { text, mode: 'streaming' }
        }));
      };

      this.ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Binary audio data
          const { metadata, audio } = this.parseBinaryFrame(event.data);
          if (onAudioChunk) onAudioChunk(audio, metadata);
        } else {
          // JSON control message
          const msg = JSON.parse(event.data);
          if (msg.type === 'progress' && onProgress) {
            onProgress(msg);
          } else if (msg.type === 'complete') {
            resolve();
          } else if (msg.type === 'error') {
            reject(msg.error);
          }
        }
      };

      this.ws.onerror = (error) => reject(error);
    });
  }

  parseBinaryFrame(buffer) {
    const view = new DataView(buffer);
    // Parse frame header
    const magic = view.getUint16(0);
    const msgType = view.getUint8(2);
    const metadataLength = view.getUint32(4);

    // Parse metadata
    const metadataBytes = new Uint8Array(buffer, 8, metadataLength);
    const metadata = JSON.parse(new TextDecoder().decode(metadataBytes));

    // Parse audio payload
    const payloadLength = view.getUint32(8 + metadataLength);
    const audio = new Int16Array(buffer, 8 + metadataLength + 4, payloadLength / 2);

    return { metadata, audio };
  }
}
```

## 12. 版本控制

协议版本通过子协议协商：

```javascript
const ws = new WebSocket("ws://host/tts", "voxcpm-tts-v1");
```

服务器响应：

```
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Protocol: voxcpm-tts-v1
```

## 13. 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.1.0 | 2025-02-18 | 添加 `voice_id` 参数支持声音克隆；更新默认值：`inference_timesteps=30`, `denoise=true`；添加 HTTP API 文档；添加环境变量配置；添加调试音频模式 |
| 1.0.0 | 2025-01-XX | 初始版本 |
