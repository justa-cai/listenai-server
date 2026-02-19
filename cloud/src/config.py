import os
from dataclasses import dataclass, field
from typing import Optional
import logging
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9400
    ping_interval: int = 30
    ping_timeout: int = 300
    max_connections: int = 100
    max_concurrent: int = 50
    interaction_timeout: int = 600
    session_timeout: int = 3600
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    log_level: str = "INFO"
    log_format: str = "json"


@dataclass
class ASRConfig:
    service_url: str = "ws://192.168.1.169:9200"
    timeout: int = 60


@dataclass
class TTSConfig:
    service_url: str = "ws://192.168.1.169:9300/tts"
    timeout: int = 300
    voice_id: str = ""
    mode: str = "streaming"
    cfg_value: float = 2.0
    inference_timesteps: int = 30


@dataclass
class LLMConfig:
    base_url: str = "http://192.168.13.228:8000/v1/"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class MCPConfig:
    enabled: bool = True
    server_name: str = "arcs-mini-mcp-server"
    server_version: str = "1.0.0"
    protocol_version: str = "2024-11-05"
    instructions: str = "ARCS Mini MCP Server"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_size: int = 512


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    @classmethod
    def from_env(cls) -> "Config":
        server = ServerConfig(
            host=os.getenv("CLOUD_HOST", "0.0.0.0"),
            port=int(os.getenv("CLOUD_PORT", "9400")),
            ping_interval=int(os.getenv("CLOUD_PING_INTERVAL", "30")),
            ping_timeout=int(os.getenv("CLOUD_PING_TIMEOUT", "300")),
            max_connections=int(os.getenv("CLOUD_MAX_CONNECTIONS", "100")),
            max_concurrent=int(os.getenv("CLOUD_MAX_CONCURRENT", "50")),
            interaction_timeout=int(os.getenv("CLOUD_INTERACTION_TIMEOUT", "600")),
            session_timeout=int(os.getenv("CLOUD_SESSION_TIMEOUT", "3600")),
            rate_limit_enabled=os.getenv("CLOUD_RATE_LIMIT_ENABLED", "true").lower()
            == "true",
            rate_limit_per_minute=int(os.getenv("CLOUD_RATE_LIMIT_PER_MINUTE", "60")),
            log_level=os.getenv("CLOUD_LOG_LEVEL", "INFO"),
            log_format=os.getenv("CLOUD_LOG_FORMAT", "json"),
        )

        asr = ASRConfig(
            service_url=os.getenv("ASR_SERVICE_URL", "ws://192.168.1.169:9200"),
            timeout=int(os.getenv("ASR_TIMEOUT", "60")),
        )

        tts = TTSConfig(
            service_url=os.getenv("TTS_SERVICE_URL", "ws://192.168.1.169:9300/tts"),
            timeout=int(os.getenv("TTS_TIMEOUT", "300")),
            voice_id=os.getenv("TTS_VOICE_ID", ""),
            mode=os.getenv("TTS_MODE", "streaming"),
            cfg_value=float(os.getenv("TTS_CFG_VALUE", "2.0")),
            inference_timesteps=int(os.getenv("TTS_INFERENCE_TIMESTEPS", "30")),
        )

        llm = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://192.168.13.228:8000/v1/"),
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("LLM_API_KEY"),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )

        mcp = MCPConfig(
            enabled=os.getenv("MCP_ENABLED", "true").lower() == "true",
            server_name=os.getenv("MCP_SERVER_NAME", "arcs-mini-mcp-server"),
            server_version=os.getenv("MCP_SERVER_VERSION", "1.0.0"),
            protocol_version=os.getenv("MCP_PROTOCOL_VERSION", "2024-11-05"),
            instructions=os.getenv("MCP_INSTRUCTIONS", "ARCS Mini MCP Server"),
        )

        audio = AudioConfig(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("AUDIO_CHANNELS", "1")),
            bits_per_sample=int(os.getenv("AUDIO_BITS_PER_SAMPLE", "16")),
            frame_size=int(os.getenv("AUDIO_FRAME_SIZE", "512")),
        )

        return cls(server=server, asr=asr, tts=tts, llm=llm, mcp=mcp, audio=audio)


def setup_logging(config: Config) -> logging.Logger:
    log_level = getattr(logging, config.server.log_level.upper())

    if config.server.log_format == "json":
        import json
        from datetime import datetime

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root_logger.handlers = [handler]
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    return logging.getLogger("cloud")
