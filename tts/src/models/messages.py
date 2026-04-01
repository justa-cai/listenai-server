"""Message data models using Pydantic."""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, validator


class TTSRequestParams(BaseModel):
    """TTS request parameters."""

    text: str
    mode: Literal["streaming", "non_streaming"] = "streaming"
    prompt_wav_url: Optional[str] = None
    prompt_text: Optional[str] = None
    cfg_value: float = Field(default=2.0, description="LM guidance value")
    inference_timesteps: int = Field(default=10, description="Number of inference timesteps")
    normalize: bool = False
    denoise: bool = False
    retry_badcase: bool = True
    retry_badcase_max_times: int = Field(default=3, ge=0, le=10)
    retry_badcase_ratio_threshold: float = Field(default=6.0, ge=1.0, le=20.0)

    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()

    @validator("cfg_value")
    def validate_cfg_value(cls, v):
        if not 0.1 <= v <= 10.0:
            raise ValueError("cfg_value must be between 0.1 and 10.0")
        return v

    @validator("inference_timesteps")
    def validate_inference_timesteps(cls, v):
        if not 1 <= v <= 50:
            raise ValueError("inference_timesteps must be between 1 and 50")
        return v


class TTSRequest(BaseModel):
    """TTS request message from client."""

    type: Literal["tts_request"]
    request_id: str
    params: TTSRequestParams


class CancelRequest(BaseModel):
    """Cancel request message from client."""

    type: Literal["cancel"]
    request_id: str


class PingMessage(BaseModel):
    """Ping message for heartbeat."""

    type: Literal["ping"]
    timestamp: int = 0


class PongMessage(BaseModel):
    """Pong response message."""

    type: Literal["pong"]
    timestamp: int
    server_time: int


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(BaseModel):
    """Error message from server."""

    type: Literal["error"] = "error"
    request_id: Optional[str] = None
    error: ErrorDetail


class ProgressMessage(BaseModel):
    """Progress update message from server."""

    type: Literal["progress"] = "progress"
    request_id: str
    state: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str


class CompleteMessage(BaseModel):
    """Complete message from server."""

    type: Literal["complete"] = "complete"
    request_id: str
    result: dict[str, Any]
