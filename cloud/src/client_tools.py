"""
客户端工具管理器

管理客户端注册的工具，支持工具注册、验证和 OpenAI 格式转换。
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClientTool:
    """客户端工具定义"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 格式

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI 工具格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


@dataclass
class PendingToolCall:
    """等待客户端结果的工具调用"""
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    future: asyncio.Future


class ClientToolManager:
    """客户端工具管理器"""

    def __init__(
        self,
        max_tools: int = 32,
        tool_timeout: int = 30,
        result_queue_size: int = 10
    ):
        self.max_tools = max_tools
        self.tool_timeout = tool_timeout
        self.result_queue_size = result_queue_size

        # 工具注册表
        self._tools: Dict[str, ClientTool] = {}

        # 等待结果的工具调用
        self._pending_calls: Dict[str, PendingToolCall] = {}

        self._lock = asyncio.Lock()

    @property
    def tool_count(self) -> int:
        """已注册的工具数量"""
        return len(self._tools)

    @property
    def tools(self) -> list[ClientTool]:
        """获取所有已注册的工具"""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[ClientTool]:
        """获取指定名称的工具"""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    async def register_tools(self, tools_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        注册客户端工具

        Args:
            tools_definitions: 工具定义列表

        Returns:
            注册结果列表
        """
        results = []
        async with self._lock:
            for tool_def in tools_definitions:
                name = tool_def.get("name")
                description = tool_def.get("description", "")
                parameters = tool_def.get("parameters", {})

                result = {
                    "name": name,
                    "status": "registered"
                }

                # 验证工具定义
                validation_error = self._validate_tool_definition(name, description, parameters)
                if validation_error:
                    result["status"] = "failed"
                    result["error"] = validation_error
                    results.append(result)
                    continue

                # 检查是否超过最大数量
                if name not in self._tools and len(self._tools) >= self.max_tools:
                    result["status"] = "failed"
                    result["error"] = f"Maximum tool limit ({self.max_tools}) reached"
                    results.append(result)
                    continue

                # 注册工具
                self._tools[name] = ClientTool(
                    name=name,
                    description=description,
                    parameters=parameters
                )
                logger.info(f"Registered client tool: {name}")
                results.append(result)

        return results

    def _validate_tool_definition(self, name: str, description: str, parameters: dict) -> Optional[str]:
        """
        验证工具定义

        Returns:
            错误信息，如果验证通过则返回 None
        """
        if not name:
            return "Tool name is required"

        if not isinstance(name, str):
            return "Tool name must be a string"

        # 检查名称格式（允许字母、数字、下划线、点号）
        # 支持带命名空间的工具名称，如 "weather.get_current" 或 "device.light.turn_on"
        import re
        # 规则：以字母或下划线开头，可包含字母、数字、下划线、点号
        # 不能以点号结尾，不能有连续的点号
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', name):
            return "Tool name must be a valid identifier (letters, numbers, underscores, dots, not starting with a number)"
        if name.endswith('.') or '..' in name:
            return "Tool name cannot end with a dot or contain consecutive dots"

        if not description:
            return "Tool description is required"

        if not isinstance(parameters, dict):
            return "Tool parameters must be an object"

        # 验证 JSON Schema 格式
        if parameters:
            schema_error = self._validate_json_schema(parameters)
            if schema_error:
                return f"Invalid parameters schema: {schema_error}"

        return None

    def _validate_json_schema(self, schema: dict) -> Optional[str]:
        """验证 JSON Schema 格式"""
        if "type" not in schema:
            return "Schema must have a 'type' field"

        if schema["type"] != "object":
            return "Root schema type must be 'object'"

        return None

    def unregister_all(self) -> None:
        """取消所有工具注册"""
        self._tools.clear()
        logger.info("All client tools unregistered")

    async def create_pending_call(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> tuple[str, asyncio.Future]:
        """
        创建一个等待客户端结果的工具调用

        Returns:
            (call_id, future)
        """
        call_id = str(uuid.uuid4())
        future = asyncio.Future()

        self._pending_calls[call_id] = PendingToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            future=future
        )

        logger.info(f"Created pending tool call: {call_id} for tool: {tool_name}")

        # 设置超时
        asyncio.create_task(self._timeout_pending_call(call_id))

        return call_id, future

    async def _timeout_pending_call(self, call_id: str) -> None:
        """超时处理"""
        await asyncio.sleep(self.tool_timeout)

        if call_id in self._pending_calls:
            pending = self._pending_calls[call_id]
            if not pending.future.done():
                pending.future.set_exception(
                    TimeoutError(f"Tool result timeout after {self.tool_timeout} seconds")
                )
                del self._pending_calls[call_id]
                logger.warning(f"Tool call timeout: {call_id}")

    async def handle_tool_result(
        self,
        call_id: str,
        result: Any,
        success: bool,
        error: Optional[str] = None
    ) -> bool:
        """
        处理客户端返回的工具结果

        Returns:
            是否成功处理（call_id 是否有效）
        """
        if call_id not in self._pending_calls:
            logger.warning(f"Received result for unknown call_id: {call_id}")
            return False

        pending = self._pending_calls[call_id]

        if pending.future.done():
            logger.warning(f"Call {call_id} already completed")
            del self._pending_calls[call_id]
            return False

        if success:
            pending.future.set_result(result)
            logger.info(f"Tool call {call_id} completed successfully")
        else:
            pending.future.set_exception(Exception(error or "Tool execution failed"))
            logger.warning(f"Tool call {call_id} failed: {error}")

        del self._pending_calls[call_id]
        return True

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """
        获取所有工具的 OpenAI 格式定义

        Returns:
            OpenAI 工具格式列表
        """
        return [tool.to_openai_tool() for tool in self._tools.values()]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        准备执行客户端工具（创建等待调用）

        注意：此方法只创建 pending call 并返回 call_id，
        实际执行由客户端完成，结果通过 handle_tool_result 返回

        Returns:
            包含 call_id 和状态的结果
        """
        if not self.has_tool(tool_name):
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }

        call_id, future = await self.create_pending_call(tool_name, arguments)

        return {
            "success": True,
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "future": future  # 用于等待结果
        }
