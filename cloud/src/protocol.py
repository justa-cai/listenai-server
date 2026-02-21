from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
from datetime import datetime


class ClientMessageType(str, Enum):
    TEXT_INPUT = "text_input"
    CONFIGURE = "configure"
    START_SESSION = "start_session"
    END_SESSION = "end_session"
    PING = "ping"


class ServerMessageType(str, Enum):
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    STATUS = "status"
    PONG = "pong"


class ErrorCode(str, Enum):
    INVALID_MESSAGE = "INVALID_MESSAGE"
    UNKNOWN_MESSAGE_TYPE = "UNKNOWN_MESSAGE_TYPE"
    LLM_ERROR = "LLM_ERROR"
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
class TextInputMessage(BaseMessage):
    type: str = ClientMessageType.TEXT_INPUT.value
    text: str = ""
    session_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["text"] = self.text
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@dataclass
class ConfigureMessage(BaseMessage):
    type: str = ClientMessageType.CONFIGURE.value
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_context: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.enable_context is not None:
            d["enable_context"] = self.enable_context
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
            return {"type": None, "error": "Binary data not supported"}

    try:
        msg = json.loads(data)
        return msg
    except json.JSONDecodeError:
        return {"type": None, "error": "Invalid JSON"}


def create_error_message(
    code: ErrorCode, message: str, details: Optional[str] = None
) -> ErrorMessage:
    return ErrorMessage(code=code.value, message=message, details=details)


def create_status_message(
    status: str, data: Optional[dict[str, Any]] = None
) -> StatusMessage:
    return StatusMessage(status=status, data=data or {})
