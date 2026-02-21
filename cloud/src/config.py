import os
from dataclasses import dataclass
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
    session_timeout: int = 3600
    log_level: str = "INFO"
    log_format: str = "json"


@dataclass
class LLMConfig:
    base_url: str = "http://192.168.13.228:8000/v1/"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = (
        "你是一个专业的语音助手助手。请遵循以下规范：\n"
        "1. 回复简洁明了，直接回答用户问题\n"
        "2. 禁止使用任何表情符号、特殊符号（如 emoji、★、◆、● 等）\n"
        "3. 使用纯文本格式，避免使用 markdown 格式（如 **加粗**、*斜体*、`代码` 等）\n"
        "4. 回复内容适合语音播报，使用自然口语化的表达\n"
        "5. 不要重复用户的问题，直接给出答案或建议"
    )
    # 是否在对话中包含上下文历史
    enable_context: bool = False


@dataclass
class MCPConfig:
    enabled: bool = True
    server_name: str = "arcs-mini-mcp-server"
    server_version: str = "1.0.0"
    protocol_version: str = "2024-11-05"
    instructions: str = "ARCS Mini MCP Server"


@dataclass
class ClientToolsConfig:
    enabled: bool = True
    max_tools: int = 32
    tool_timeout: int = 30
    result_queue_size: int = 10


@dataclass
class Config:
    server: ServerConfig
    llm: LLMConfig
    mcp: MCPConfig
    client_tools: ClientToolsConfig

    @classmethod
    def from_env(cls) -> "Config":
        server = ServerConfig(
            host=os.getenv("CLOUD_HOST", "0.0.0.0"),
            port=int(os.getenv("CLOUD_PORT", "9400")),
            ping_interval=int(os.getenv("CLOUD_PING_INTERVAL", "30")),
            ping_timeout=int(os.getenv("CLOUD_PING_TIMEOUT", "300")),
            max_connections=int(os.getenv("CLOUD_MAX_CONNECTIONS", "100")),
            session_timeout=int(os.getenv("CLOUD_SESSION_TIMEOUT", "3600")),
            log_level=os.getenv("CLOUD_LOG_LEVEL", "INFO"),
            log_format=os.getenv("CLOUD_LOG_FORMAT", "json"),
        )

        llm = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://192.168.13.228:8000/v1/"),
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("LLM_API_KEY"),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            system_prompt=os.getenv(
                "LLM_SYSTEM_PROMPT",
                "你是一个专业的语音助手助手。请遵循以下规范：\n"
                "1. 回复简洁明了，直接回答用户问题\n"
                "2. 禁止使用任何表情符号、特殊符号（如 emoji、★、◆、● 等）\n"
                "3. 使用纯文本格式，避免使用 markdown 格式（如 **加粗**、*斜体*、`代码` 等）\n"
                "4. 回复内容适合语音播报，使用自然口语化的表达\n"
                "5. 不要重复用户的问题，直接给出答案或建议",
            ),
            enable_context=os.getenv("LLM_ENABLE_CONTEXT", "false").lower() == "true",
        )

        mcp = MCPConfig(
            enabled=os.getenv("MCP_ENABLED", "true").lower() == "true",
            server_name=os.getenv("MCP_SERVER_NAME", "arcs-mini-mcp-server"),
            server_version=os.getenv("MCP_SERVER_VERSION", "1.0.0"),
            protocol_version=os.getenv("MCP_PROTOCOL_VERSION", "2024-11-05"),
            instructions=os.getenv("MCP_INSTRUCTIONS", "ARCS Mini MCP Server"),
        )

        client_tools = ClientToolsConfig(
            enabled=os.getenv("CLIENT_TOOLS_ENABLED", "true").lower() == "true",
            max_tools=int(os.getenv("CLIENT_TOOLS_MAX_COUNT", "32")),
            tool_timeout=int(os.getenv("CLIENT_TOOL_TIMEOUT", "30")),
            result_queue_size=int(os.getenv("CLIENT_TOOL_RESULT_QUEUE_SIZE", "10")),
        )

        return cls(server=server, llm=llm, mcp=mcp, client_tools=client_tools)


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
