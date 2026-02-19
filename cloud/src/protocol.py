from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
from datetime import datetime


class ClientMessageType(str, Enum):
    AUDIO_DATA = "audio_data"
    CONFIGURE = "configure"
    START_SESSION = "start_session"
    END_SESSION = "end_session"
    PING = "ping"


class ServerMessageType(str, Enum):
    ASR_RESULT = "asr_result"
    LLM_RESPONSE = "llm_response"
    TTS_AUDIO = "tts_audio"
    ERROR = "error"
    STATUS = "status"
    PONG = "pong"
    TOOL_CALL = "tool_call"


class ErrorCode(str, Enum):
    INVALID_MESSAGE = "INVALID_MESSAGE"
    UNKNOWN_MESSAGE_TYPE = "UNKNOWN_MESSAGE_TYPE"
    ASR_ERROR = "ASR_ERROR"
    LLM_ERROR = "LLM_ERROR"
    TTS_ERROR = "TTS_ERROR"
    SESSION_ERROR = "SESSION_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class BaseMessage:
    type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "timestamp": self.timestamp}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AudioDataMessage(BaseMessage):
    type: str = ClientMessageType.AUDIO_DATA.value
    data: bytes = b""


@dataclass
class ConfigureMessage(BaseMessage):
    type: str = ClientMessageType.CONFIGURE.value
    language: Optional[str] = None
    voice_id: Optional[str] = None
    sample_rate: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.language:
            d["language"] = self.language
        if self.voice_id:
            d["voice_id"] = self.voice_id
        if self.sample_rate:
            d["sample_rate"] = self.sample_rate
        return d


@dataclass
class StartSessionMessage(BaseMessage):
    type: str = ClientMessageType.START_SESSION.value
    session_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@dataclass
class EndSessionMessage(BaseMessage):
    type: str = ClientMessageType.END_SESSION.value


@dataclass
class PingMessage(BaseMessage):
    type: str = ClientMessageType.PING.value


@dataclass
class PongMessage(BaseMessage):
    type: str = ServerMessageType.PONG.value


@dataclass
class ASRResultMessage(BaseMessage):
    type: str = ServerMessageType.ASR_RESULT.value
    text: str = ""
    is_final: bool = False
    segment_id: Optional[str] = None
    is_speaking: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["text"] = self.text
        d["is_final"] = self.is_final
        d["is_speaking"] = self.is_speaking
        if self.segment_id:
            d["segment_id"] = self.segment_id
        return d


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class LLMResponseMessage(BaseMessage):
    type: str = ServerMessageType.LLM_RESPONSE.value
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_final: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["content"] = self.content
        d["is_final"] = self.is_final
        if self.tool_calls:
            d["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id}
                for tc in self.tool_calls
            ]
        return d


@dataclass
class TTSAudioMessage(BaseMessage):
    type: str = ServerMessageType.TTS_AUDIO.value
    data: bytes = b""
    is_final: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["is_final"] = self.is_final
        return d


@dataclass
class ErrorMessage(BaseMessage):
    type: str = ServerMessageType.ERROR.value
    code: str = ErrorCode.INTERNAL_ERROR.value
    message: str = ""
    details: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["code"] = self.code
        d["message"] = self.message
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class StatusMessage(BaseMessage):
    type: str = ServerMessageType.STATUS.value
    status: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["status"] = self.status
        d["data"] = self.data
        return d


@dataclass
class ToolCallMessage(BaseMessage):
    type: str = ServerMessageType.TOOL_CALL.value
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = True
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["tool_name"] = self.tool_name
        d["arguments"] = self.arguments
        d["result"] = self.result
        d["success"] = self.success
        d["duration_ms"] = self.duration_ms
        return d


def parse_client_message(data: str | bytes) -> dict[str, Any]:
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": ClientMessageType.AUDIO_DATA.value, "data": data}

    try:
        msg = json.loads(data)
        return msg
    except json.JSONDecodeError:
        return {"type": ClientMessageType.AUDIO_DATA.value, "data": data}


def create_error_message(
    code: ErrorCode, message: str, details: Optional[str] = None
) -> ErrorMessage:
    return ErrorMessage(code=code.value, message=message, details=details)


def create_status_message(
    status: str, data: Optional[dict[str, Any]] = None
) -> StatusMessage:
    return StatusMessage(status=status, data=data or {})
