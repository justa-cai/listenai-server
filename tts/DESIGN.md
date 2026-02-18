# VoxCPM WebSocket TTS Server - 设计文档

## 1. 系统架构

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                           Client Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Web App │  │  Mobile  │  │ Desktop  │  │  CLI     │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼────────────┼─────────────┼─────────────┼──────────────┘
        │            │              │              │
        └──────────────────────────────────────────┘
                            │
                    WebSocket (TCP)
                            │
┌─────────────────────────────────────────────────────────────────┐
│                      Server Layer                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  WebSocket Server                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │ │
│  │  │ Connection  │  │   Message   │  │   Session   │       │ │
│  │  │  Manager    │  │  Handler    │  │  Manager    │       │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │ │
│  └─────────┼────────────────┼──────────────────┼─────────────┘ │
│            │                │                  │               │
│  ┌─────────┴────────────────┴──────────────────┴─────────────┐ │
│  │                    Request Router                         │ │
│  └──────────────────────────────┬────────────────────────────┘ │
└─────────────────────────────────┼──────────────────────────────┘
                                  │
┌─────────────────────────────────┼──────────────────────────────┐
│                    Business Logic Layer                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    TTS Service                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │   Request   │  │   Stream    │  │   Result    │      │ │
│  │  │  Validator  │  │  Generator  │  │  Processor  │      │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │ │
│  └─────────┼────────────────┼──────────────────┼────────────┘ │
│            │                │                  │              │
│  ┌─────────┴────────────────┴──────────────────┴────────────┐ │
│  │                    Task Queue                            │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │ │
│  │  │Task │ │Task │ │Task │ │Task │ │Task │ │Task │        │ │
│  │  │  1  │ │  2  │ │  3  │ │  4  │ │  5  │ │  6  │        │ │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘        │ │
│  └─────┼──────┼──────┼──────┼──────┼──────┼───────────────┘ │
└────────┼──────┼──────┼──────┼──────┼──────┼────────────────┘
         │      │      │      │      │      │
         │      │      │      └──────┴──────┘
         │      │      │              │
         │      │      │      ┌───────┴────────┐
         │      │      │      │                │
┌────────┴──────┴──────┴──────┴┐                │
│          Model Layer         │                │
│  ┌─────────────────────────┐ │                │
│  │    VoxCPM Model Wrapper │ │                │
│  │  ┌─────────────────┐    │ │                │
│  │  │  VoxCPM-0.5B    │    │ │                │
│  │  │     Model       │    │ │                │
│  │  └────────┬────────┘    │ │                │
│  │           │              │ │                │
│  │  ┌────────┴────────┐    │ │                │
│  │  │  Model Cache    │    │ │                │
│  │  └─────────────────┘    │ │                │
│  └─────────────────────────┘ │                │
└──────────────────────────────┴────────────────┘
```

### 1.2 组件说明

#### 1.2.1 Server Layer (服务器层)

| 组件 | 职责 |
|------|------|
| WebSocket Server | 处理 WebSocket 连接、消息收发 |
| Connection Manager | 管理客户端连接生命周期 |
| Message Handler | 解析和处理客户端消息 |
| Session Manager | 管理会话状态和请求上下文 |
| Request Router | 将请求路由到对应的处理器 |

#### 1.2.2 Business Logic Layer (业务逻辑层)

| 组件 | 职责 |
|------|------|
| TTS Service | TTS 核心业务逻辑 |
| Request Validator | 验证请求参数 |
| Stream Generator | 生成流式音频 |
| Result Processor | 处理生成结果 |
| Task Queue | 管理并发任务队列 |

#### 1.2.3 Model Layer (模型层)

| 组件 | 职责 |
|------|------|
| VoxCPM Model Wrapper | 封装 VoxCPM 模型接口 |
| Model Cache | 模型缓存管理 |

## 2. 详细设计

### 2.1 WebSocket Server 设计

```python
# src/server.py

import asyncio
import websockets
from typing import Set
from dataclasses import dataclass
from .connection import Connection
from .session import SessionManager
from .message_handler import MessageHandler
from .config import ServerConfig

@dataclass
class ServerState:
    connections: Set[Connection]
    active_requests: dict[str, asyncio.Task]
    request_queue: asyncio.Queue

class VoxCPMWebSocketServer:
    """VoxCPM WebSocket TTS Server"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.state = ServerState(
            connections=set(),
            active_requests={},
            request_queue=asyncio.Queue(maxsize=config.max_queue_size)
        )
        self.session_manager = SessionManager()
        self.message_handler = MessageHandler(
            model_config=config.model,
            queue=self.state.request_queue
        )

    async def serve(self, host: str, port: int):
        """启动服务器"""
        async with websockets.serve(
            self._handle_connection,
            host,
            port,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
            max_size=self.config.max_message_size,
            compression=None
        ) as server:
            await server.serve_forever()

    async def _handle_connection(
        self,
        websocket: websockets.WebSocketServerProtocol,
        path: str
    ):
        """处理新连接"""
        connection = Connection(websocket)
        self.state.connections.add(connection)

        try:
            await self._handle_messages(connection)
        except websockets.ConnectionClosed:
            pass
        finally:
            await self._cleanup_connection(connection)

    async def _handle_messages(self, connection: Connection):
        """处理连接消息"""
        async for raw_message in connection.websocket:
            try:
                if isinstance(raw_message, bytes):
                    # 二进制消息（暂不支持客户端发送二进制）
                    await connection.send_error(
                        "UNSUPPORTED_FORMAT",
                        "Binary messages from client are not supported"
                    )
                    continue

                # JSON 消息
                message = json.loads(raw_message)
                response = await self.message_handler.handle(
                    connection,
                    message,
                    self.session_manager
                )

                if response:
                    await connection.send(response)

            except json.JSONDecodeError:
                await connection.send_error("INVALID_JSON", "Invalid JSON format")
            except Exception as e:
                await connection.send_error("INTERNAL_ERROR", str(e))

    async def _cleanup_connection(self, connection: Connection):
        """清理连接"""
        self.state.connections.discard(connection)

        # 取消该连接的所有活动请求
        for request_id, task in list(self.state.active_requests.items()):
            if task.get_name() == connection.id:
                task.cancel()

        await connection.close()
```

### 2.2 连接管理设计

```python
# src/connection.py

import uuid
import websockets
from datetime import datetime
from typing import Optional

class Connection:
    """WebSocket 连接封装"""

    def __init__(self, websocket: websockets.WebSocketServerProtocol):
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.created_at = datetime.now()
        self.last_ping = datetime.now()
        self.metadata = {}

    async def send(self, data: str | bytes):
        """发送消息"""
        if isinstance(data, str):
            await self.websocket.send(data)
        else:
            await self.websocket.send(data)

    async def send_json(self, obj: dict):
        """发送 JSON 消息"""
        await self.send(json.dumps(obj))

    async def send_binary_frame(
        self,
        msg_type: int,
        metadata: dict,
        audio_data: bytes
    ):
        """发送二进制帧"""
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata_json)

        frame = bytearray()
        # Magic
        frame.extend([0xAA, 0x55])
        # Message Type
        frame.append(msg_type)
        # Reserved
        frame.append(0x00)
        # Metadata Length
        frame.extend(metadata_length.to_bytes(4, 'big'))
        # Metadata
        frame.extend(metadata_json)
        # Payload Length
        frame.extend(len(audio_data).to_bytes(4, 'big'))
        # Payload
        frame.extend(audio_data)

        await self.send(bytes(frame))

    async def send_progress(
        self,
        request_id: str,
        state: str,
        progress: float,
        message: str
    ):
        """发送进度更新"""
        await self.send_json({
            "type": "progress",
            "request_id": request_id,
            "state": state,
            "progress": progress,
            "message": message
        })

    async def send_complete(
        self,
        request_id: str,
        result: dict
    ):
        """发送完成消息"""
        await self.send_json({
            "type": "complete",
            "request_id": request_id,
            "result": result
        })

    async def send_error(
        self,
        code: str,
        message: str,
        details: dict = None
    ):
        """发送错误消息"""
        await self.send_json({
            "type": "error",
            "request_id": self.metadata.get("current_request_id"),
            "error": {
                "code": code,
                "message": message,
                "details": details or {}
            }
        })

    async def close(self):
        """关闭连接"""
        await self.websocket.close()
```

### 2.3 消息处理设计

```python
# src/message_handler.py

from typing import Optional
from .connection import Connection
from .session import SessionManager
from .tasks import TTSRequestTask, CancelTask
from .validators import TTSRequestValidator

class MessageHandler:
    """消息处理器"""

    def __init__(self, model_config: dict, queue: asyncio.Queue):
        self.model_config = model_config
        self.queue = queue
        self.validator = TTSRequestValidator()

    async def handle(
        self,
        connection: Connection,
        message: dict,
        session_manager: SessionManager
    ) -> Optional[str]:
        """处理消息"""
        msg_type = message.get("type")

        if msg_type == "tts_request":
            return await self._handle_tts_request(
                connection, message, session_manager
            )
        elif msg_type == "cancel":
            return await self._handle_cancel(
                connection, message, session_manager
            )
        elif msg_type == "ping":
            return await self._handle_ping(connection, message)
        else:
            await connection.send_error(
                "UNKNOWN_MESSAGE_TYPE",
                f"Unknown message type: {msg_type}"
            )

    async def _handle_tts_request(
        self,
        connection: Connection,
        message: dict,
        session_manager: SessionManager
    ) -> Optional[str]:
        """处理 TTS 请求"""
        request_id = message.get("request_id")
        params = message.get("params", {})

        # 记录当前请求 ID（用于错误处理）
        connection.metadata["current_request_id"] = request_id

        # 验证请求
        validation_result = self.validator.validate(params)
        if not validation_result.is_valid:
            await connection.send_error(
                "INVALID_PARAMS",
                validation_result.error_message,
                validation_result.errors
            )
            return None

        # 创建会话
        session = session_manager.create_session(
            request_id=request_id,
            connection_id=connection.id,
            params=params
        )

        # 创建任务
        task = TTSRequestTask(
            session=session,
            model_config=self.model_config,
            queue=self.queue
        )

        # 启动任务
        asyncio.create_task(task.run())

        return None

    async def _handle_cancel(
        self,
        connection: Connection,
        message: dict,
        session_manager: SessionManager
    ) -> Optional[str]:
        """处理取消请求"""
        request_id = message.get("request_id")

        session = session_manager.get_session(request_id)
        if session and session.task:
            session.task.cancel()

        await connection.send_complete(request_id, {
            "cancelled": True
        })

        return None

    async def _handle_ping(
        self,
        connection: Connection,
        message: dict
    ) -> str:
        """处理心跳"""
        timestamp = message.get("timestamp", 0)
        return json.dumps({
            "type": "pong",
            "timestamp": timestamp,
            "server_time": int(time.time())
        })
```

### 2.4 TTS 任务处理设计

```python
# src/tasks.py

import asyncio
import numpy as np
from voxcpm import VoxCPM
from .session import Session
from .audio_processor import AudioProcessor
from .config import ModelConfig

class TTSRequestTask:
    """TTS 请求任务"""

    def __init__(
        self,
        session: Session,
        model_config: ModelConfig,
        queue: asyncio.Queue
    ):
        self.session = session
        self.model_config = model_config
        self.queue = queue
        self.audio_processor = AudioProcessor()
        self._cancelled = False

    async def run(self):
        """运行任务"""
        try:
            await self.queue.put(self)

            # 更新状态
            await self._send_progress("processing", 0.0, "Processing request")

            # 获取模型
            model = await self._get_model()

            # 执行 TTS
            params = self.session.params
            if params.get("mode") == "streaming":
                await self._generate_streaming(model, params)
            else:
                await self._generate_non_streaming(model, params)

            # 发送完成消息
            await self._send_complete()

        except asyncio.CancelledError:
            self._cancelled = True
            await self._send_complete(cancelled=True)
        except Exception as e:
            await self._send_error(str(e))

    async def _get_model(self) -> VoxCPM:
        """获取模型实例"""
        # 从模型缓存获取
        from .model_cache import ModelCache
        return await ModelCache.get_instance(self.model_config)

    async def _generate_streaming(
        self,
        model: VoxCPM,
        params: dict
    ):
        """生成流式音频"""
        await self._send_progress("generating", 0.1, "Generating audio...")

        sequence = 0
        total_samples = 0
        chunks = []

        for chunk in model.generate_streaming(
            text=params["text"],
            prompt_wav_path=params.get("prompt_wav_path"),
            prompt_text=params.get("prompt_text"),
            cfg_value=params.get("cfg_value", 2.0),
            inference_timesteps=params.get("inference_timesteps", 10),
            normalize=params.get("normalize", False),
            denoise=params.get("denoise", False),
            retry_badcase=params.get("retry_badcase", True),
            retry_badcase_max_times=params.get("retry_badcase_max_times", 3),
            retry_badcase_ratio_threshold=params.get(
                "retry_badcase_ratio_threshold", 6.0
            )
        ):
            if self._cancelled:
                break

            # 转换为 PCM 16-bit
            audio_bytes = self.audio_processor.to_pcm16(
                chunk,
                model.tts_model.sample_rate
            )

            # 发送音频块
            await self._send_audio_chunk(
                sequence,
                audio_bytes,
                model.tts_model.sample_rate,
                is_final=False
            )

            chunks.append(chunk)
            total_samples += len(chunk)
            sequence += 1

        self.session.result = {
            "duration": total_samples / model.tts_model.sample_rate,
            "sample_rate": model.tts_model.sample_rate,
            "samples": total_samples,
            "chunks": sequence
        }

    async def _generate_non_streaming(
        self,
        model: VoxCPM,
        params: dict
    ):
        """生成非流式音频"""
        await self._send_progress("generating", 0.5, "Generating audio...")

        wav = model.generate(
            text=params["text"],
            prompt_wav_path=params.get("prompt_wav_path"),
            prompt_text=params.get("prompt_text"),
            cfg_value=params.get("cfg_value", 2.0),
            inference_timesteps=params.get("inference_timesteps", 10),
            normalize=params.get("normalize", False),
            denoise=params.get("denoise", False),
            retry_badcase=params.get("retry_badcase", True),
            retry_badcase_max_times=params.get("retry_badcase_max_times", 3),
            retry_badcase_ratio_threshold=params.get(
                "retry_badcase_ratio_threshold", 6.0
            )
        )

        # 转换为 PCM 16-bit
        audio_bytes = self.audio_processor.to_pcm16(
            wav,
            model.tts_model.sample_rate
        )

        await self._send_progress("encoding", 0.9, "Encoding audio...")

        # 发送完整音频
        await self._send_audio_full(
            audio_bytes,
            model.tts_model.sample_rate,
            len(wav) / model.tts_model.sample_rate
        )

        self.session.result = {
            "duration": len(wav) / model.tts_model.sample_rate,
            "sample_rate": model.tts_model.sample_rate,
            "samples": len(wav),
            "chunks": 1
        }

    async def _send_progress(
        self,
        state: str,
        progress: float,
        message: str
    ):
        """发送进度更新"""
        connection = self.session.connection
        await connection.send_progress(
            self.session.request_id,
            state,
            progress,
            message
        )

    async def _send_audio_chunk(
        self,
        sequence: int,
        audio_bytes: bytes,
        sample_rate: int,
        is_final: bool
    ):
        """发送音频块"""
        connection = self.session.connection
        await connection.send_binary_frame(
            msg_type=0x01,  # STREAMING_CHUNK
            metadata={
                "request_id": self.session.request_id,
                "sequence": sequence,
                "sample_rate": sample_rate,
                "is_final": is_final
            },
            audio_data=audio_bytes
        )

    async def _send_audio_full(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        duration: float
    ):
        """发送完整音频"""
        connection = self.session.connection
        await connection.send_binary_frame(
            msg_type=0x02,  # NON_STREAMING
            metadata={
                "request_id": self.session.request_id,
                "sample_rate": sample_rate,
                "duration": duration
            },
            audio_data=audio_bytes
        )

    async def _send_complete(self, cancelled: bool = False):
        """发送完成消息"""
        connection = self.session.connection
        result = self.session.result or {"cancelled": cancelled}
        await connection.send_complete(self.session.request_id, result)

    async def _send_error(self, message: str):
        """发送错误消息"""
        connection = self.session.connection
        await connection.send_error("GENERATION_FAILED", message)

    def cancel(self):
        """取消任务"""
        self._cancelled = True
```

### 2.5 模型缓存设计

```python
# src/model_cache.py

from voxcpm import VoxCPM
from typing import Optional
import asyncio

class ModelCache:
    """模型缓存管理"""

    _instance: Optional['ModelCache'] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._models: dict[str, VoxCPM] = {}
        self._loading: dict[str, asyncio.Lock] = {}

    @classmethod
    async def get_instance(
        cls,
        config: 'ModelConfig'
    ) -> VoxCPM:
        """获取模型实例（单例）"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return await cls._instance._get_model(config)

    async def _get_model(self, config: 'ModelConfig') -> VoxCPM:
        """获取或加载模型"""
        model_key = config.model_name

        # 检查是否已加载
        if model_key in self._models:
            return self._models[model_key]

        # 检查是否正在加载
        if model_key in self._loading:
            await self._loading[model_key].acquire()
            self._loading[model_key].release()
            return self._models[model_key]

        # 开始加载
        self._loading[model_key] = asyncio.Lock()
        await self._loading[model_key].acquire()

        try:
            # 在线程池中加载模型（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: VoxCPM.from_pretrained(config.model_name)
            )

            self._models[model_key] = model
            return model
        finally:
            self._loading[model_key].release()
            del self._loading[model_key]
```

### 2.6 会话管理设计

```python
# src/session.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Session:
    """TTS 会话"""
    request_id: str
    connection_id: str
    params: dict
    created_at: datetime = field(default_factory=datetime.now)
    task: Optional['TTSRequestTask'] = None
    result: Optional[dict] = None

class SessionManager:
    """会话管理器"""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(
        self,
        request_id: str,
        connection_id: str,
        params: dict
    ) -> Session:
        """创建会话"""
        session = Session(
            request_id=request_id,
            connection_id=connection_id,
            params=params
        )
        self._sessions[request_id] = session
        return session

    def get_session(self, request_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(request_id)

    def remove_session(self, request_id: str):
        """移除会话"""
        self._sessions.pop(request_id, None)
```

### 2.7 参数验证设计

```python
# src/validators.py

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    error_message: str = ""
    errors: Dict[str, str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = {}

class TTSRequestValidator:
    """TTS 请求参数验证器"""

    # 参数约束配置
    CONSTRAINTS = {
        "max_text_length": 5000,
        "cfg_value_range": (0.1, 10.0),
        "inference_timesteps_range": (1, 50),
        "retry_badcase_max_times_range": (0, 10),
        "retry_badcase_ratio_threshold_range": (1.0, 20.0)
    }

    def validate(self, params: dict) -> ValidationResult:
        """验证参数"""
        errors = {}

        # 验证 text
        if "text" not in params:
            errors["text"] = "is required"
        elif not isinstance(params["text"], str):
            errors["text"] = "must be a string"
        elif len(params["text"]) == 0:
            errors["text"] = "cannot be empty"
        elif len(params["text"]) > self.CONSTRAINTS["max_text_length"]:
            errors["text"] = (
                f"too long (max {self.CONSTRAINTS['max_text_length']} characters)"
            )

        # 验证 mode
        if "mode" in params and params["mode"] not in ("streaming", "non_streaming"):
            errors["mode"] = "must be 'streaming' or 'non_streaming'"

        # 验证 cfg_value
        if "cfg_value" in params:
            cfg_value = params["cfg_value"]
            min_val, max_val = self.CONSTRAINTS["cfg_value_range"]
            if not isinstance(cfg_value, (int, float)):
                errors["cfg_value"] = "must be a number"
            elif cfg_value < min_val or cfg_value > max_val:
                errors["cfg_value"] = f"must be between {min_val} and {max_val}"

        # 验证 inference_timesteps
        if "inference_timesteps" in params:
            steps = params["inference_timesteps"]
            min_val, max_val = self.CONSTRAINTS["inference_timesteps_range"]
            if not isinstance(steps, int):
                errors["inference_timesteps"] = "must be an integer"
            elif steps < min_val or steps > max_val:
                errors["inference_timesteps"] = f"must be between {min_val} and {max_val}"

        # 验证布尔值参数
        for param in ("normalize", "denoise", "retry_badcase"):
            if param in params and not isinstance(params[param], bool):
                errors[param] = "must be a boolean"

        # 验证 retry_badcase_max_times
        if "retry_badcase_max_times" in params:
            max_times = params["retry_badcase_max_times"]
            min_val, max_val = self.CONSTRAINTS["retry_badcase_max_times_range"]
            if not isinstance(max_times, int):
                errors["retry_badcase_max_times"] = "must be an integer"
            elif max_times < min_val or max_times > max_val:
                errors["retry_badcase_max_times"] = f"must be between {min_val} and {max_val}"

        # 生成结果
        if errors:
            error_message = "Validation failed: " + ", ".join(
                f"{k}: {v}" for k, v in errors.items()
            )
            return ValidationResult(is_valid=False, error_message=error_message, errors=errors)

        return ValidationResult(is_valid=True)
```

### 2.8 音频处理设计

```python
# src/audio_processor.py

import numpy as np

class AudioProcessor:
    """音频处理器"""

    def to_pcm16(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """
        将音频转换为 PCM 16-bit 格式

        Args:
            audio: numpy 音频数组
            sample_rate: 采样率

        Returns:
            PCM 16-bit 字节数据
        """
        # 归一化到 [-1, 1]
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio, -1.0, 1.0)
        elif audio.dtype == np.int16:
            return audio.tobytes()
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / (2**31)
        else:
            raise ValueError(f"Unsupported audio dtype: {audio.dtype}")

        # 转换为 int16
        audio_int16 = (audio * 32767).astype(np.int16)

        return audio_int16.tobytes()

    def from_pcm16(self, data: bytes) -> np.ndarray:
        """
        从 PCM 16-bit 字节数据转换为音频数组

        Args:
            data: PCM 16-bit 字节数据

        Returns:
            numpy 音频数组
        """
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
```

### 2.9 配置管理设计

```python
# src/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8765
    ping_interval: int = 30
    ping_timeout: int = 10
    max_message_size: int = 2**20  # 1MB
    max_connections: int = 100
    max_concurrent_requests: int = 10
    max_queue_size: int = 50

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "openbmb/VoxCPM-0.5B"
    device: str = "cuda"  # or "cpu"
    cache_dir: Optional[str] = None

@dataclass
class Config:
    """总配置"""
    server: ServerConfig
    model: ModelConfig

    @classmethod
    def from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        import os

        server = ServerConfig(
            host=os.getenv("TTS_HOST", "0.0.0.0"),
            port=int(os.getenv("TTS_PORT", "8765")),
            max_connections=int(os.getenv("TTS_MAX_CONNECTIONS", "100")),
            max_concurrent_requests=int(os.getenv("TTS_MAX_CONCURRENT", "10"))
        )

        model = ModelConfig(
            model_name=os.getenv("TTS_MODEL_NAME", "openbmb/VoxCPM-0.5B"),
            device=os.getenv("TTS_DEVICE", "cuda")
        )

        return cls(server=server, model=model)
```

## 3. 数据模型

### 3.1 消息数据模型

```python
# src/models/messages.py

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class TTSRequestParams(BaseModel):
    """TTS 请求参数"""
    text: str
    mode: Literal["streaming", "non_streaming"] = "streaming"
    prompt_wav_url: Optional[str] = None
    prompt_text: Optional[str] = None
    cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    inference_timesteps: int = Field(default=10, ge=1, le=50)
    normalize: bool = False
    denoise: bool = False
    retry_badcase: bool = True
    retry_badcase_max_times: int = Field(default=3, ge=0, le=10)
    retry_badcase_ratio_threshold: float = Field(default=6.0, ge=1.0, le=20.0)

class TTSRequest(BaseModel):
    """TTS 请求消息"""
    type: Literal["tts_request"]
    request_id: str
    params: TTSRequestParams

class CancelRequest(BaseModel):
    """取消请求消息"""
    type: Literal["cancel"]
    request_id: str

class ProgressMessage(BaseModel):
    """进度消息"""
    type: Literal["progress"]
    request_id: str
    state: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str

class CompleteMessage(BaseModel):
    """完成消息"""
    type: Literal["complete"]
    request_id: str
    result: dict

class ErrorMessage(BaseModel):
    """错误消息"""
    type: Literal["error"]
    request_id: Optional[str]
    error: ErrorDetail

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str
    message: str
    details: dict = {}
```

## 4. 并发控制

### 4.1 任务队列设计

```python
# src/queue.py

import asyncio
from typing import Optional
from .tasks import TTSRequestTask

class TaskQueue:
    """任务队列（带并发控制）"""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue[TTSRequestTask] = asyncio.Queue()
        self._running_tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def put(self, task: TTSRequestTask):
        """添加任务到队列"""
        await self._queue.put(task)

    async def worker(self):
        """工作协程"""
        while True:
            # 获取任务
            task = await self._queue.get()

            # 创建工作协程
            async def run_task():
                async with self._semaphore:
                    try:
                        await task.run()
                    finally:
                        self._queue.task_done()

            worker_task = asyncio.create_task(run_task())
            self._running_tasks.add(worker_task)
            worker_task.add_done_callback(self._running_tasks.discard)

    async def start(self, num_workers: int = 1):
        """启动工作协程"""
        for _ in range(num_workers):
            asyncio.create_task(self.worker())

    @property
    def pending_count(self) -> int:
        """待处理任务数"""
        return self._queue.qsize()

    @property
    def running_count(self) -> int:
        """正在运行的任务数"""
        return len(self._running_tasks)
```

## 5. 错误处理

### 5.1 错误类型定义

```python
# src/errors.py

class VoxCPMError(Exception):
    """VoxCPM 基础错误"""
    code: str = "INTERNAL_ERROR"
    http_status: int = 500

class ValidationError(VoxCPMError):
    """参数验证错误"""
    code = "INVALID_PARAMS"
    http_status = 400

class ModelNotLoadedError(VoxCPMError):
    """模型未加载错误"""
    code = "MODEL_NOT_LOADED"
    http_status = 503

class GenerationFailedError(VoxCPMError):
    """生成失败错误"""
    code = "GENERATION_FAILED"
    http_status = 500

class RateLimitError(VoxCPMError):
    """速率限制错误"""
    code = "RATE_LIMITED"
    http_status = 429
```

### 5.2 错误处理装饰器

```python
# src/decorators.py

import functools
import logging
from .errors import VoxCPMError
from .connection import Connection

logger = logging.getLogger(__name__)

def handle_errors(connection: Connection):
    """错误处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except VoxCPMError as e:
                await connection.send_error(e.code, str(e))
            except Exception as e:
                logger.exception("Unexpected error")
                await connection.send_error("INTERNAL_ERROR", str(e))
        return wrapper
    return decorator
```

## 6. 监控和日志

### 6.1 指标收集

```python
# src/metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time

# 请求计数
request_counter = Counter(
    'voxcpm_tts_requests_total',
    'Total TTS requests',
    ['mode', 'status']
)

# 请求延迟
request_duration = Histogram(
    'voxcpm_tts_request_duration_seconds',
    'TTS request duration',
    ['mode']
)

# 活动连接数
active_connections = Gauge(
    'voxcpm_tts_active_connections',
    'Active WebSocket connections'
)

# 队列长度
queue_length = Gauge(
    'voxcpm_tts_queue_length',
    'Pending requests in queue'
)

# 正在运行的任务数
running_tasks = Gauge(
    'voxcpm_tts_running_tasks',
    'Currently running TTS tasks'
)
```

### 6.2 结构化日志

```python
# src/logging.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging(level: str = "INFO"):
    """设置日志"""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
```

## 7. 部署设计

### 7.1 环境变量配置

```bash
# .env.example

# 服务器配置
TTS_HOST=0.0.0.0
TTS_PORT=8765
TTS_MAX_CONNECTIONS=100
TTS_MAX_CONCURRENT=10

# 模型配置
TTS_MODEL_NAME=openbmb/VoxCPM-0.5B
TTS_DEVICE=cuda

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json

# 监控配置
METRICS_ENABLED=true
METRICS_PORT=9090
```

## 8. 测试设计

### 8.1 单元测试

```python
# tests/test_validators.py

import pytest
from src.validators import TTSRequestValidator

def test_validate_success():
    validator = TTSRequestValidator()
    params = {
        "text": "Hello, world!",
        "mode": "streaming"
    }
    result = validator.validate(params)
    assert result.is_valid

def test_validate_missing_text():
    validator = TTSRequestValidator()
    params = {"mode": "streaming"}
    result = validator.validate(params)
    assert not result.is_valid
    assert "text" in result.errors

def test_validate_text_too_long():
    validator = TTSRequestValidator()
    params = {
        "text": "a" * 10000,
        "mode": "streaming"
    }
    result = validator.validate(params)
    assert not result.is_valid
    assert "too long" in result.errors["text"]
```

### 8.2 集成测试

```python
# tests/test_integration.py

import pytest
import asyncio
import websockets
from src.server import VoxCPMWebSocketServer

@pytest.fixture
async def server():
    config = ServerConfig(port=8765)
    server = VoxCPMWebSocketServer(config)

    async def run_server():
        await server.serve("localhost", 8765)

    task = asyncio.create_task(run_server())
    yield server
    task.cancel()

@pytest.mark.asyncio
async def test_tts_request_streaming(server):
    uri = "ws://localhost:8765/tts"
    async with websockets.connect(uri) as ws:
        request = {
            "type": "tts_request",
            "request_id": "test-001",
            "params": {
                "text": "Hello, world!",
                "mode": "streaming"
            }
        }
        await ws.send(json.dumps(request))

        # 接收音频块
        chunks_received = 0
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                chunks_received += 1
            else:
                msg = json.loads(message)
                if msg["type"] == "complete":
                    break

        assert chunks_received > 0
```

## 9. 项目目录结构

```
voxcpm-tts-server/
├── src/
│   ├── __init__.py
│   ├── main.py                 # 入口文件
│   ├── server.py               # WebSocket 服务器
│   ├── connection.py           # 连接管理
│   ├── session.py              # 会话管理
│   ├── message_handler.py      # 消息处理
│   ├── tasks.py                # TTS 任务
│   ├── model_cache.py          # 模型缓存
│   ├── validators.py           # 参数验证
│   ├── audio_processor.py      # 音频处理
│   ├── queue.py                # 任务队列
│   ├── errors.py               # 错误定义
│   ├── decorators.py           # 装饰器
│   ├── metrics.py              # 监控指标
│   └── config.py               # 配置管理
├── tests/
│   ├── __init__.py
│   ├── test_validators.py
│   ├── test_audio_processor.py
│   └── test_integration.py
├── config/
│   ├── prometheus.yml
│   └── model_config.json
├── docs/
│   ├── PROTOCOL.md
│   ├── STORY.md
│   └── DESIGN.md
├── scripts/
│   ├── start.sh
│   └── stop.sh
├── requirements.txt
└── .env.example
```

## 10. 性能优化

### 10.1 优化策略

| 优化项 | 策略 | 预期效果 |
|--------|------|----------|
| 模型加载 | 预加载 + 单例 | 减少首次请求延迟 |
| 音频处理 | 异步处理 | 提高吞吐量 |
| 连接管理 | 连接复用 | 减少握手开销 |
| 内存管理 | 及时释放 | 避免内存泄漏 |
| 并发控制 | 任务队列 | 平衡负载 |

### 10.2 缓存策略

```python
# src/cache.py

from functools import lru_cache
from typing import Optional
import hashlib

class AudioCache:
    """音频结果缓存（可选）"""

    def __init__(self, max_size: int = 100):
        self._cache: dict[str, bytes] = {}
        self._max_size = max_size

    def _make_key(self, text: str, params: dict) -> str:
        """生成缓存键"""
        data = f"{text}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, text: str, params: dict) -> Optional[bytes]:
        """获取缓存"""
        key = self._make_key(text, params)
        return self._cache.get(key)

    def put(self, text: str, params: dict, audio: bytes):
        """存入缓存"""
        if len(self._cache) >= self._max_size:
            # 移除最旧的缓存项
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        key = self._make_key(text, params)
        self._cache[key] = audio
```
