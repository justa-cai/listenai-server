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
    log_format: str = "console"  # Options: json, console, text


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
    # 高德地图 API 配置
    amap_api_key: Optional[str] = None
    weather_api_enabled: bool = True


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
            amap_api_key=os.getenv("AMAP_API_KEY"),
            weather_api_enabled=os.getenv("WEATHER_API_ENABLED", "true").lower() == "true",
        )

        client_tools = ClientToolsConfig(
            enabled=os.getenv("CLIENT_TOOLS_ENABLED", "true").lower() == "true",
            max_tools=int(os.getenv("CLIENT_TOOLS_MAX_COUNT", "32")),
            tool_timeout=int(os.getenv("CLIENT_TOOL_TIMEOUT", "30")),
            result_queue_size=int(os.getenv("CLIENT_TOOL_RESULT_QUEUE_SIZE", "10")),
        )

        return cls(server=server, llm=llm, mcp=mcp, client_tools=client_tools)


class ColoredFormatter(logging.Formatter):
    """带颜色的高亮日志格式化器"""

    # ANSI 颜色码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    BOLD = '\033[1m'
    RESET = '\033[0m'
    GRAY = '\033[90m'
    BLUE = '\033[34m'

    def format(self, record):
        # 获取颜色
        level_color = self.COLORS.get(record.levelname, self.RESET)

        # 格式化各部分
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        level = f"{level_color}{self.BOLD}{record.levelname:<8}{self.RESET}"
        logger_name = f"{self.BLUE}{record.name}{self.RESET}"
        location = f"{self.GRAY}{record.module}:{record.funcName}:{record.lineno}{self.RESET}"
        message = record.getMessage()

        # 组合格式化输出
        formatted = f"{timestamp} | {level} | {logger_name} | {location} | {message}"

        # 处理异常信息
        if record.exc_info:
            formatted += '\n' + self.formatException(record.exc_info)

        return formatted


class JsonFormatter(logging.Formatter):
    """JSON 格式日志格式化器"""

    def format(self, record):
        import json
        from datetime import datetime

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


def setup_logging(config: Config) -> logging.Logger:
    log_level = getattr(logging, config.server.log_level.upper())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    handler = logging.StreamHandler()

    # 根据配置选择格式化器
    log_format = config.server.log_format.lower()

    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    elif log_format == "console":
        handler.setFormatter(ColoredFormatter())
    else:  # text
        handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s')
        )

    root_logger.handlers = [handler]

    return logging.getLogger("cloud")
