# ASR WebSocket API 文档

## 概述

ASR WebSocket 服务提供实时语音识别功能，基于 FunASR-Nano 模型，支持中文语音识别。服务采用双向通信模式，客户端持续发送音频流，服务端实时返回识别结果。

## 连接信息

| 参数 | 值 |
|------|-----|
| WebSocket 地址 | `ws://<host>:9200/` |
| 默认主机 | `0.0.0.0` |
| 默认端口 | `9200` |
| 连接超时 | 60 秒 |
| Ping 间隔 | 20 秒 |

## 音频格式要求

| 参数 | 值 |
|------|-----|
| 编码格式 | PCM Int16 |
| 采样率 | 16000 Hz |
| 声道 | 单声道 (Mono) |
| 字节序 | 小端序 (Little Endian) |

**建议发送间隔**: 20ms (640 字节)

## 工作模式

服务支持两种工作模式：

### 流式模式（默认）

实时语音识别，服务端通过 VAD 自动检测语音起止点。

- 适用场景：实时对讲、语音助手
- 特点：低延迟、自动分段、持续识别

### 批量模式

非流式识别，客户端上传完整音频后显式触发识别。

- 适用场景：录音文件转录、音频处理
- 特点：一次性识别完整音频、无 VAD 干预

## 消息协议

### 客户端 → 服务端

#### 1. 音频数据 (二进制消息)

直接发送 PCM 音频数据的二进制字节流。

```python
# 示例：发送音频数据
await websocket.send(audio_bytes)  # audio_bytes 为 bytes 类型
```

#### 2. 控制命令 (JSON 文本消息)

##### Ping 命令

用于检测连接状态：

```json
{
  "command": "ping"
}
```

**响应**:
```json
{
  "type": "pong"
}
```

##### Reset 命令

重置服务端内部状态（VAD、缓冲区等）：

```json
{
  "command": "reset"
}
```

**响应**:
```json
{
  "type": "reset",
  "message": "State reset successfully"
}
```

##### Speaker_Register_Start 命令

开始说话人注册模式，采集用户声纹：

```json
{
  "command": "speaker_register_start",
  "user_id": "user_001"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `command` | string | 是 | 固定为 `"speaker_register_start"` |
| `user_id` | string | 是 | 用户唯一标识符 |

**响应**:
```json
{
  "type": "speaker_register_progress",
  "state": "started",
  "message": "Registration started for user: user_001, please send audio (min 3.0s)"
}
```

**说明**:
- 发送此命令后进入注册模式，客户端需要发送至少 3 秒的音频数据
- 音频时长建议 3-30 秒
- 音频数据通过二进制消息持续发送
- 发送完成后调用 `speaker_register_end` 命令

##### Speaker_Register_End 命令

结束说话人注册，提取声纹特征并保存：

```json
{
  "command": "speaker_register_end"
}
```

**响应（成功）**:
```json
{
  "type": "speaker_register_result",
  "success": true,
  "user_id": "user_001",
  "duration": 5.236,
  "timestamp": 1704067200000
}
```

**响应（失败）**:
```json
{
  "type": "error",
  "message": "Audio too short: 2.1s < 3.0s",
  "code": 13,
  "timestamp": 1704067200000
}
```

##### Speaker_Verify_Enable 命令

启用说话人验证功能：

```json
{
  "command": "speaker_verify_enable",
  "user_id": "user_001",
  "required": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `command` | string | 是 | 固定为 `"speaker_verify_enable"` |
| `user_id` | string | 是 | 要验证的目标用户 ID |
| `required` | boolean | 否 | 是否强制验证（验证失败时跳过 ASR），默认 true |

**响应**:
```json
{
  "type": "speaker_verify_enabled",
  "user_id": "user_001",
  "required": true,
  "timestamp": 1704067200000
}
```

**说明**:
- 启用后，每个语音段识别前会先验证说话人身份
- 如果 `required=true` 且验证失败，不会返回 ASR 结果
- 验证结果通过 `speaker_verify_result` 消息返回

##### Speaker_Verify_Disable 命令

禁用说话人验证功能：

```json
{
  "command": "speaker_verify_disable"
}
```

**响应**:
```json
{
  "type": "speaker_verify_disabled",
  "timestamp": 1704067200000
}
```

##### Speaker_List 命令

获取所有已注册的说话人列表：

```json
{
  "command": "speaker_list"
}
```

**响应**:
```json
{
  "type": "speaker_list",
  "speakers": [
    {
      "user_id": "user_001",
      "dimension": 192,
      "registered_at": 1704067200
    },
    {
      "user_id": "user_002",
      "dimension": 192,
      "registered_at": 1704067300
    }
  ],
  "count": 2,
  "timestamp": 1704067200000
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `speakers` | array | 说话人列表 |
| `speakers[].user_id` | string | 用户 ID |
| `speakers[].dimension` | number | 声纹向量维度 |
| `speakers[].registered_at` | number | 注册时间戳（秒） |
| `count` | number | 说话人总数 |

##### Batch_Start 命令

开始批量识别模式，准备接收完整音频：

```json
{
  "command": "batch_start",
  "sample_rate": 16000,
  "language": "中文"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `command` | string | 是 | 固定为 `"batch_start"` |
| `sample_rate` | number | 否 | 采样率，默认 16000 |
| `language` | string | 否 | 识别语言，默认 `"中文"` |

**响应**:
```json
{
  "type": "batch_progress",
  "state": "receiving",
  "message": "Batch mode started, waiting for audio data"
}
```

##### Batch_End 命令

结束音频接收并触发识别：

```json
{
  "command": "batch_end"
}
```

**响应**:
```json
{
  "type": "batch_progress",
  "state": "recognizing",
  "message": "Processing audio..."
}
```

随后会返回 `batch_result` 消息。

### 服务端 → 客户端

#### VAD 事件消息

语音活动检测事件，当检测到语音开始或结束时发送：

```json
{
  "type": "vad",
  "event": "speech_start",
  "timestamp": 1704067200000
}
```

语音开始事件（`speech_start`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"vad"` |
| `event` | string | 事件类型：`"speech_start"` 或 `"speech_end"` |
| `timestamp` | number | 时间戳（毫秒） |

语音结束事件（`speech_end`）：

```json
{
  "type": "vad",
  "event": "speech_end",
  "duration": 2.456,
  "samples": 39296,
  "timestamp": 1704067202456
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"vad"` |
| `event` | string | 事件类型：`"speech_start"` 或 `"speech_end"` |
| `duration` | number | 语音段时长（秒），仅 `speech_end` 事件包含 |
| `samples` | number | 语音段采样点数，仅 `speech_end` 事件包含 |
| `timestamp` | number | 时间戳（毫秒） |

#### 识别结果消息

```json
{
  "type": "result",
  "text": "识别的文本内容",
  "is_final": true,
  "is_speeching": false,
  "timestamp": 1704067200000,
  "segment_id": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"result"` |
| `text` | string | 识别的文本内容 |
| `is_final` | boolean | 是否为最终结果（服务端目前只发送最终结果） |
| `is_speeching` | boolean | 当前是否正在检测到语音 |
| `timestamp` | number | 时间戳（毫秒） |
| `segment_id` | number | 语音段序号，从 1 开始递增 |

#### 错误消息

```json
{
  "type": "error",
  "message": "错误描述",
  "code": 1,
  "timestamp": 1704067200000
}
```

#### 批量识别结果消息

```json
{
  "type": "batch_result",
  "text": "识别的完整文本内容",
  "duration": 5.236,
  "samples": 83776,
  "timestamp": 1704067200000
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"batch_result"` |
| `text` | string | 识别的文本内容 |
| `duration` | number | 音频时长（秒） |
| `samples` | number | 音频采样点数 |
| `timestamp` | number | 时间戳（毫秒） |

#### 批量识别进度消息

```json
{
  "type": "batch_progress",
  "state": "receiving",
  "received_bytes": 64000,
  "message": "Receiving audio data..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"batch_progress"` |
| `state` | string | 状态：`receiving`(接收中)、`recognizing`(识别中)、`complete`(完成) |
| `received_bytes` | number | 已接收字节数（仅在 `receiving` 状态时包含） |
| `message` | string | 状态描述 |

#### 说话人验证结果消息

```json
{
  "type": "speaker_verify_result",
  "user_id": "user_001",
  "success": true,
  "score": 0.8532,
  "threshold": 0.3,
  "timestamp": 1704067200000
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"speaker_verify_result"` |
| `user_id` | string | 验证的用户 ID |
| `success` | boolean | 验证是否成功（相似度 >= 阈值） |
| `score` | number | 相似度分数（0-1） |
| `threshold` | number | 验证阈值 |

#### 说话人注册进度消息

```json
{
  "type": "speaker_register_progress",
  "state": "receiving",
  "received_bytes": 64000,
  "duration": 4.0,
  "message": "Collecting audio...",
  "timestamp": 1704067200000
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"speaker_register_progress"` |
| `state` | string | 状态：`started`(已开始)、`receiving`(接收中) |
| `received_bytes` | number | 已接收字节数 |
| `duration` | number | 已接收音频时长（秒） |
| `message` | string | 状态描述 |

#### 说话人注册结果消息

```json
{
  "type": "speaker_register_result",
  "success": true,
  "user_id": "user_001",
  "duration": 5.236,
  "timestamp": 1704067200000
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 消息类型，固定为 `"speaker_register_result"` |
| `success` | boolean | 注册是否成功 |
| `user_id` | string | 注册的用户 ID |
| `duration` | number | 音频时长（秒） |

| 错误码 | 说明 |
|--------|------|
| 1 | 无效的 JSON 消息 |
| 2 | 处理异常 |
| 3 | 未知命令 |
| 4 | 批量模式未开始 |
| 5 | 音频数据为空 |
| 6 | 音频时长过短 |
| 10 | 缺少 user_id 参数（注册） |
| 11 | 注册模式未激活 |
| 12 | 未收到音频数据（注册） |
| 13 | 注册音频过短（< 3秒） |
| 14 | 声纹特征提取失败 |
| 15 | 缺少 user_id 参数（验证） |
| 16 | 目标说话人未注册 |

## 处理流程

### 流式模式（默认）

```
┌─────────┐                    ┌──────────────┐
│ Client  │                    │    Server     │
└────┬────┘                    └──────┬───────┘
     │                               │
     │ ──── WebSocket Connect ────>  │
     │                               │
     │ ──── Audio Bytes ──────────>  │
     │ ──── Audio Bytes ──────────>  │
     │                               │
     │ <──── VAD: speech_start ──────│
     │                               │
     │ ──── Audio Bytes ──────────>  │
     │ ──── Audio Bytes ──────────>  │
     │ ...                           │
     │                               │
     │ <──── VAD: speech_end ────────│
     │                               │
     │ <──── Result (is_final=true) ─│
     │                               │
     │ ──── Audio Bytes ──────────>  │
     │ ...                           │
     └───────────────────────────────┘
```

### 批量模式

```
┌─────────┐                    ┌──────────────┐
│ Client  │                    │    Server     │
└────┬────┘                    └──────┬───────┘
     │                               │
     │ ──── WebSocket Connect ────>  │
     │                               │
     │ ──── batch_start ─────────>  │ 进入批量模式
     │                               │
     │ <──── batch_progress ─────────│ state: receiving
     │                               │
     │ ──── Audio Bytes ──────────>  │
     │ ──── Audio Bytes ──────────>  │ 缓存音频
     │ ──── Audio Bytes ──────────>  │
     │                               │
     │ ──── batch_end ────────────>  │ 开始识别
     │                               │
     │ <──── batch_progress ─────────│ state: recognizing
     │                               │
     │ <──── batch_result ───────────│ 返回识别结果
     │                               │
     └───────────────────────────────┘
```

### 说话人注册流程

```
┌─────────┐                    ┌──────────────┐
│ Client  │                    │    Server     │
└────┬────┘                    └──────┬───────┘
     │                               │
     │ ──── WebSocket Connect ────>  │
     │                               │
     │ ──── speaker_register_start ─>  │ 进入注册模式
     │     (user_id: user_001)        │
     │                               │
     │ <──── speaker_register_progress│ state: started
     │                               │
     │ ──── Audio Bytes ──────────>  │ 收集音频（3-30秒）
     │ ──── Audio Bytes ──────────>  │
     │ ──── Audio Bytes ──────────>  │
     │ ...                           │
     │ <──── speaker_register_progress│ state: receiving (duration)
     │                               │
     │ ──── speaker_register_end ────>  │ 提取声纹并保存
     │                               │
     │ <──── speaker_register_result─│ success: true, user_id: user_001
     │                               │
     └───────────────────────────────┘
```

### 说话人验证流程

```
┌─────────┐                    ┌──────────────┐
│ Client  │                    │    Server     │
└────┬────┘                    └──────┬───────┘
     │                               │
     │ ──── WebSocket Connect ────>  │
     │                               │
     │ ──── speaker_verify_enable ──>  │ 启用验证
     │     (user_id: user_001)        │     (user_id: user_001, required: true)
     │                               │
     │ <──── speaker_verify_enabled ──│ 确认已启用
     │                               │
     │ ──── Audio Bytes ──────────>  │
     │ ──── Audio Bytes ──────────>  │
     │                               │
     │ <──── VAD: speech_end ────────│
     │                               │
     │ <──── speaker_verify_result ───│ success: true, score: 0.85
     │                               │
     │ <──── Result (is_final=true) ─│ 验证通过，返回 ASR 结果
     │                               │
     └───────────────────────────────┘
```

**验证失败时（required=true）**：
```
     │ <──── speaker_verify_result ───│ success: false, score: 0.25
     │                               │
     │ (无 ASR 结果)                   │ 验证失败，跳过 ASR
```

### 消息时序说明

1. 客户端发送连续的音频数据流
2. 服务端检测到语音开始时，发送 `speech_start` 事件
3. 语音结束后，发送 `speech_end` 事件（包含时长和采样点数）
4. 随后发送识别结果（如果通过过滤条件）

## 音频处理机制

服务端采用多层过滤机制确保识别质量：

### 1. 语音活动检测 (VAD)

- **进入语音状态**: 需要连续 3 帧语音帧
- **退出语音状态**: 需要连续 5 帧静音帧（迟滞机制）

### 2. 能量过滤

- 阈值: `0.01`
- 低于阈值的音频段将被跳过

### 3. 语音比例过滤

- 阈值: `30%`
- 语音帧占比低于阈值的音频段被视为噪声

### 4. 文本验证

- 最小文本长度: 2 个字符
- 必须包含中文或英文字符（过滤纯标点符号）

## 降噪功能

服务端支持 RNNoise 降噪（默认关闭）：

| 参数 | 值 |
|------|-----|
| 状态 | 默认关闭 |
| 采样率 | 16000 Hz |
| 延迟 | < 10ms |

如需启用，请修改服务端配置 `NS_ENABLED = True`。

## 识别限制

| 限制项 | 值 |
|--------|-----|
| 最小语音段时长 | 0.3 秒 |
| 最大缓冲时长 | 2.0 秒 |
| 支持语言 | 中文 |

## 客户端实现示例

### JavaScript/TypeScript

```javascript
class ASRClient {
  constructor(url = 'ws://localhost:9200') {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.resultCallback = null;
    this.errorCallback = null;

    this.ws.onopen = () => {
      console.log('ASR WebSocket connected');
      // 启动心跳
      this.heartbeatInterval = setInterval(() => {
        this.sendCommand('ping');
      }, 30000);
    };

    this.ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
      clearInterval(this.heartbeatInterval);
    };
  }

  handleMessage(message) {
    switch (message.type) {
      case 'vad':
        if (this.vadCallback) {
          this.vadCallback(message);
        }
        if (message.event === 'speech_start') {
          console.log('语音开始');
        } else if (message.event === 'speech_end') {
          console.log(`语音结束，时长: ${message.duration}s`);
        }
        break;
      case 'result':
        if (this.resultCallback) {
          this.resultCallback(message);
        }
        break;
      case 'error':
        if (this.errorCallback) {
          this.errorCallback(message);
        }
        console.error('ASR Error:', message.message);
        break;
      case 'pong':
        console.log('Pong received');
        break;
    }
  }

  sendAudio(int16Array) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(int16Array.buffer);
    }
  }

  sendCommand(command) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command }));
    }
  }

  reset() {
    this.sendCommand('reset');
  }

  onResult(callback) {
    this.resultCallback = callback;
  }

  onVad(callback) {
    this.vadCallback = callback;
  }

  onError(callback) {
    this.errorCallback = callback;
  }

  close() {
    clearInterval(this.heartbeatInterval);
    this.ws.close();
  }
}

// 使用示例
const client = new ASRClient('ws://192.168.1.100:9200');

// 监听 VAD 事件
client.onVad((event) => {
  if (event.event === 'speech_start') {
    console.log('检测到语音开始');
    // 可以在这里显示录音指示灯
  } else if (event.event === 'speech_end') {
    console.log(`检测到语音结束，时长: ${event.duration}秒`);
    // 隐藏录音指示灯
  }
});

client.onResult((result) => {
  console.log(`识别结果: ${result.text}`);
  if (result.is_final) {
    console.log('最终结果');
  }
});

// 发送音频数据
// audioData 是 Int16Array
client.sendAudio(audioData);
```

### Python

```python
import asyncio
import websockets
import json
import numpy as np

async def asr_client(url="ws://localhost:9200"):
    async with websockets.connect(url) as websocket:
        print("Connected to ASR server")

        # 启动心跳任务
        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                await websocket.send(json.dumps({"command": "ping"}))

        heartbeat_task = asyncio.create_task(heartbeat())

        try:
            # 接收消息
            while True:
                message = await websocket.recv()

                if isinstance(message, bytes):
                    # 二进制消息（服务端不发送二进制，但可以处理）
                    pass
                else:
                    # JSON 消息
                    data = json.loads(message)
                    if data["type"] == "vad":
                        event = data["event"]
                        if event == "speech_start":
                            print("语音开始")
                        elif event == "speech_end":
                            duration = data.get("duration", 0)
                            samples = data.get("samples", 0)
                            print(f"语音结束，时长: {duration}秒, 采样点: {samples}")
                    elif data["type"] == "result":
                        print(f"识别结果: {data['text']}")
                        if data.get("is_final"):
                            print("最终结果")
                    elif data["type"] == "error":
                        print(f"错误: {data['message']} (code: {data['code']})")
                    elif data["type"] == "pong":
                        print("Pong")

        finally:
            heartbeat_task.cancel()

# 使用示例
asyncio.run(asr_client())
```

### 浏览器录音 + ASR 示例

```javascript
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    const asrClient = new ASRClient('ws://localhost:9200');

    processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      const pcmData = new Int16Array(inputData.length);

      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
      }

      asrClient.sendAudio(pcmData);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    asrClient.onResult = (result) => {
      console.log('识别结果:', result.text);
    };
  });
```

### 批量识别示例 (JavaScript)

```javascript
class ASRBatchClient {
  constructor(url = 'ws://localhost:9200') {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.audioBuffer = [];

    this.ws.onopen = () => {
      console.log('ASR WebSocket connected');
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
  }

  handleMessage(message) {
    switch (message.type) {
      case 'batch_progress':
        console.log(`批量识别状态: ${message.state}`);
        if (message.state === 'receiving' && message.received_bytes) {
          console.log(`已接收: ${message.received_bytes} 字节`);
        }
        break;
      case 'batch_result':
        console.log(`识别结果: ${message.text}`);
        console.log(`音频时长: ${message.duration}秒`);
        if (this.onResult) {
          this.onResult(message);
        }
        break;
      case 'error':
        console.error('错误:', message.message);
        if (this.onError) {
          this.onError(message);
        }
        break;
    }
  }

  // 发送完整音频数据进行识别
  async recognize(audioData) {
    return new Promise((resolve, reject) => {
      this.onResult = resolve;
      this.onError = reject;

      // 开始批量模式
      this.ws.send(JSON.stringify({ command: 'batch_start' }));

      // 发送音频数据（可以分块）
      const chunkSize = 4096; // 每块 4KB
      for (let i = 0; i < audioData.byteLength; i += chunkSize) {
        const chunk = audioData.slice(i, Math.min(i + chunkSize, audioData.byteLength));
        this.ws.send(chunk);
      }

      // 结束并触发识别
      this.ws.send(JSON.stringify({ command: 'batch_end' }));
    });
  }

  close() {
    this.ws.close();
  }
}

// 使用示例：识别 ArrayBuffer 音频数据
const client = new ASRBatchClient('ws://localhost:9200');

// 假设 audioArrayBuffer 是从文件读取或录音得到的音频数据
const audioArrayBuffer = /* ... */;

try {
  const result = await client.recognize(audioArrayBuffer);
  console.log('识别完成:', result.text);
} catch (error) {
  console.error('识别失败:', error);
}
```

### 批量识别示例 (Python)

```python
import asyncio
import websockets
import json
from pathlib import Path

async def batch_asr(audio_file: str, url: str = "ws://localhost:9200"):
    """批量识别音频文件"""
    async with websockets.connect(url) as ws:
        # 读取音频文件
        audio_data = Path(audio_file).read_bytes()

        print(f"开始批量识别，文件大小: {len(audio_data)} 字节")

        # 发送 batch_start 命令
        await ws.send(json.dumps({"command": "batch_start"}))

        # 接收确认消息
        response = json.loads(await ws.recv())
        print(f"服务端响应: {response}")

        # 分块发送音频数据（避免单个消息过大）
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await ws.send(chunk)

        # 发送 batch_end 命令
        await ws.send(json.dumps({"command": "batch_end"}))
        print("音频发送完成，等待识别结果...")

        # 接收识别进度和结果
        while True:
            message = await ws.recv()
            data = json.loads(message)

            if data["type"] == "batch_progress":
                print(f"状态: {data['state']}")
                if "received_bytes" in data:
                    print(f"已接收: {data['received_bytes']} 字节")
            elif data["type"] == "batch_result":
                print(f"\n识别结果: {data['text']}")
                print(f"音频时长: {data['duration']:.2f} 秒")
                print(f"采样点数: {data['samples']}")
                return data["text"]
            elif data["type"] == "error":
                print(f"错误: {data['message']}")
                return None

# 使用示例
asyncio.run(batch_asr("test_audio.pcm"))
```

## 错误处理

### 连接错误

```javascript
ws.onerror = (error) => {
  console.error('WebSocket connection error:', error);
  // 尝试重连
  setTimeout(() => reconnect(), 3000);
};
```

### 业务错误

```javascript
// 服务端返回的错误消息
{
  "type": "error",
  "message": "Invalid JSON message",
  "code": 1,
  "timestamp": 1704067200000
}
```

## 性能建议

1. **发送频率**: 建议每 20ms 发送一次音频数据 (640 字节)
2. **缓冲区**: 客户端可适当缓冲音频数据，减少网络请求次数
3. **重连机制**: 实现自动重连，网络恢复后重新建立连接
4. **心跳保活**: 定期发送 ping 命令保持连接活跃

## 配置参数

服务端可配置参数（需修改服务端代码）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `WS_HOST` | `"0.0.0.0"` | WebSocket 监听地址 |
| `WS_PORT` | `9200` | WebSocket 监听端口 |
| `VAD_THRESHOLD` | `0.5` | VAD 检测阈值 |
| `VAD_SPEECH_FRAMES` | `3` | 进入语音状态所需连续语音帧数 |
| `VAD_SILENCE_FRAMES` | `10` | 退出语音状态所需连续静音帧数 |
| `ENERGY_THRESHOLD` | `0.01` | 能量过滤阈值 |
| `VAD_SPEECH_RATIO_THRESHOLD` | `0.3` | 语音比例阈值 |
| `NS_ENABLED` | `false` | 是否启用降噪 |
| `ASR_MIN_TEXT_LENGTH` | `2` | 最小文本长度 |
| **说话人验证配置** | | |
| `SPEAKER_MODEL_ID` | `"iic/speech_eres2netv2_sv_zh-cn_16k-common"` | ModelScope 说话人验证模型 ID |
| `SPEAKER_THRESHOLD` | `0.3` | 说话人相似度阈值（0-1，越高越严格） |
| `SPEAKER_DB_PATH` | `"data/speakers.json"` | 声纹数据库文件路径 |
| `SPEAKER_MIN_REGISTRATION_DURATION` | `3.0` | 最短注册音频时长（秒） |
| `SPEAKER_MAX_REGISTRATION_DURATION` | `30.0` | 最长注册音频时长（秒） |
| `SPEAKER_VERIFICATION_REQUIRED` | `false` | 全局强制启用说话人验证 |
| `SPEAKER_VERIFICATION_DEFAULT_USER` | `""` | 全局验证的默认用户 ID |

**说话人验证说明**：
- 当 `SPEAKER_VERIFICATION_REQUIRED=true` 时，服务器启动时加载验证模型
- 当 `SPEAKER_VERIFICATION_REQUIRED=false` 时，模型延迟加载（客户端首次使用时加载）
- 建议根据实际需求调整 `SPEAKER_THRESHOLD`：
  - 更严格的验证：设置更高阈值（如 0.5-0.7）
  - 更宽松的验证：设置更低阈值（如 0.2-0.3）
  - 相似度分数使用余弦相似度，范围 0-1，越接近 1 表示越相似

## HTTP 服务

服务端同时提供 HTTP 静态文件服务：

| 参数 | 值 |
|------|-----|
| 地址 | `http://<host>:9201/` |
| 默认端口 | `9201` |

可用于访问测试页面或其他静态资源。

## 附录

### VAD 事件类型

| 事件 | 说明 | 包含字段 |
|------|------|----------|
| `speech_start` | 检测到语音开始 | `type`, `event`, `timestamp` |
| `speech_end` | 检测到语音结束 | `type`, `event`, `duration`, `samples`, `timestamp` |

### 状态码

| 状态 | 说明 |
|------|------|
| `is_speeching: true` | 正在检测到语音 |
| `is_speeching: false` | 当前静音 |
| `is_final: true` | 最终识别结果 |
| `is_final: false` | 临时结果（当前未使用） |

### 注意事项

**通用注意事项:**
1. 音频必须是 16kHz 采样率的 PCM Int16 格式
2. 建议实现客户端重连机制处理网络中断
3. 发送过短的音频（< 0.3秒）会被忽略
4. 定期发送 ping 命令保持连接活跃

**流式模式注意事项:**
1. VAD 事件与识别结果的关系：
   - `speech_start` 表示检测到语音段开始
   - `speech_end` 表示检测到语音段结束
   - 识别结果（`type: "result"`）会在 `speech_end` 之后返回
   - 如果语音段未通过质量过滤，可能只有 VAD 事件而没有识别结果
2. 服务端不支持流式临时结果，只在语音段结束时返回最终结果
3. VAD 采用双向迟滞机制：需要连续 3 帧语音才触发 `speech_start`，连续 5 帧静音才触发 `speech_end`

**批量模式注意事项:**
1. 必须先发送 `batch_start` 命令才能开始发送音频数据
2. 音频数据可以分块发送，适合大文件传输
3. 发送完所有音频数据后，必须发送 `batch_end` 命令触发识别
4. 批量模式不经过 VAD 和质量过滤，直接对完整音频进行识别
5. 批量识别期间发送的音频数据会被缓存，不会触发实时识别
6. 可以通过发送 `reset` 命令取消当前的批量识别并重置状态

**说话人验证注意事项:**
1. 说话人验证需要在 ASR 识别前先进行声纹比对
2. 启用验证后，每个语音段会先验证说话人身份，然后才执行 ASR
3. 如果 `required=true` 且验证失败，不会返回 ASR 识别结果
4. 验证相似度使用余弦相似度，范围 0-1，阈值默认为 0.3
5. 注册时建议录音 5-10 秒，声音清晰、无背景噪音
6. 同一用户多次注册会覆盖之前的声纹数据
7. 说话人验证模型（ERes2NetV2）首次使用时会自动下载（约 70MB）
8. 如果不需要说话人验证功能，保持 `SPEAKER_VERIFICATION_REQUIRED=false` 可以节省显存
