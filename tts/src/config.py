"""Configuration management for VoxCPM TTS Server."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 9300
    ping_interval: int = 30
    ping_timeout: int = 300  # Increased to 5 minutes for long TTS generation
    max_message_size: int = 2**20  # 1MB
    max_connections: int = 100
    max_concurrent_requests: int = 10
    max_queue_size: int = 50
    num_model_workers: int = 1  # Number of model worker threads
    request_timeout: int = 600  # Increased to 10 minutes

    # Audio settings
    chunk_size: int = 4096  # samples per chunk
    sample_rate: int = 16000

    # Voice cloning settings
    voice_dir: str = "voice_clone"  # Directory containing voice samples

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60

    # Monitoring
    metrics_enabled: bool = False
    metrics_port: int = 9090

    # Logging
    log_level: str = "INFO"
    log_format: str = "text"  # or "json"

    # Debug
    debug_audio: bool = False  # Enable debug audio saving to ./tmp/debug_audio/


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "VoxCPM-0.5B"  # Can be local path or HuggingFace model ID
    device: str = "cuda"  # or "cpu"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True

    @property
    def is_local_model(self) -> bool:
        """Check if the model is a local path."""
        import os
        return os.path.exists(self.model_name)

    # Default TTS parameters
    default_cfg_value: float = 2.0
    default_inference_timesteps: int = 30
    default_normalize: bool = False
    default_denoise: bool = True
    default_retry_badcase: bool = True
    default_retry_badcase_max_times: int = 3
    default_retry_badcase_ratio_threshold: float = 6.0

    # Constraints
    max_text_length: int = 5000
    cfg_value_range: tuple = (0.1, 10.0)
    inference_timesteps_range: tuple = (1, 50)
    retry_badcase_max_times_range: tuple = (0, 10)
    retry_badcase_ratio_threshold_range: tuple = (1.0, 20.0)


@dataclass
class Config:
    """Main configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        server = ServerConfig(
            host=os.getenv("TTS_HOST", "0.0.0.0"),
            port=int(os.getenv("TTS_PORT", "9300")),
            ping_interval=int(os.getenv("TTS_PING_INTERVAL", "30")),
            ping_timeout=int(os.getenv("TTS_PING_TIMEOUT", "300")),
            max_connections=int(os.getenv("TTS_MAX_CONNECTIONS", "100")),
            max_concurrent_requests=int(os.getenv("TTS_MAX_CONCURRENT", "10")),
            max_queue_size=int(os.getenv("TTS_MAX_QUEUE_SIZE", "50")),
            num_model_workers=int(os.getenv("TTS_NUM_MODEL_WORKERS", "3")),
            request_timeout=int(os.getenv("TTS_REQUEST_TIMEOUT", "600")),
            chunk_size=int(os.getenv("TTS_CHUNK_SIZE", "4096")),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "16000")),
            voice_dir=os.getenv("TTS_VOICE_DIR", "voice_clone"),
            rate_limit_enabled=os.getenv("TTS_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_per_minute=int(os.getenv("TTS_RATE_LIMIT_PER_MINUTE", "60")),
            metrics_enabled=os.getenv("TTS_METRICS_ENABLED", "false").lower() == "true",
            metrics_port=int(os.getenv("TTS_METRICS_PORT", "9090")),
            log_level=os.getenv("TTS_LOG_LEVEL", "INFO"),
            log_format=os.getenv("TTS_LOG_FORMAT", "text"),
            debug_audio=os.getenv("TTS_DEBUG_AUDIO", "false").lower() == "true",
        )

        model = ModelConfig(
            model_name=os.getenv("TTS_MODEL_NAME", "VoxCPM-0.5B"),
            device=os.getenv("TTS_DEVICE", "cuda"),
            cache_dir=os.getenv("TTS_MODEL_CACHE_DIR"),
            trust_remote_code=os.getenv("TTS_TRUST_REMOTE_CODE", "true").lower() == "true",
            default_cfg_value=float(os.getenv("TTS_DEFAULT_CFG_VALUE", "2.0")),
            default_inference_timesteps=int(os.getenv("TTS_DEFAULT_INFERENCE_TIMESTEPS", "10")),
            default_normalize=os.getenv("TTS_DEFAULT_NORMALIZE", "false").lower() == "true",
            default_denoise=os.getenv("TTS_DEFAULT_DENOISE", "false").lower() == "true",
            default_retry_badcase=os.getenv("TTS_DEFAULT_RETRY_BADCASE", "true").lower() == "true",
            default_retry_badcase_max_times=int(os.getenv("TTS_DEFAULT_RETRY_MAX_TIMES", "3")),
            default_retry_badcase_ratio_threshold=float(os.getenv("TTS_DEFAULT_RETRY_RATIO_THRESHOLD", "6.0")),
            max_text_length=int(os.getenv("TTS_MAX_TEXT_LENGTH", "5000")),
        )

        return cls(server=server, model=model)
