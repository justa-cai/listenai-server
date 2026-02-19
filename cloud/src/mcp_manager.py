import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[..., Any]] = None

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MCPToolManager:
    def __init__(
        self,
        server_name: str = "arcs-mini-mcp-server",
        server_version: str = "1.0.0",
        protocol_version: str = "2024-11-05",
        instructions: str = "",
    ):
        self.server_name = server_name
        self.server_version = server_version
        self.protocol_version = protocol_version
        self.instructions = instructions
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_call_history: list[dict[str, Any]] = []
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        self.register_tool(
            ToolDefinition(
                name="get_weather",
                description="Get weather information for a specified city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, e.g. Beijing, Shanghai",
                        }
                    },
                    "required": ["city"],
                },
                handler=self._get_weather,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="set_volume",
                description="Set device volume (0-100 range)",
                parameters={
                    "type": "object",
                    "properties": {
                        "volume": {
                            "type": "integer",
                            "description": "Volume level (0-100)",
                            "minimum": 0,
                            "maximum": 100,
                        }
                    },
                    "required": ["volume"],
                },
                handler=self._set_volume,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="play_music",
                description="Play music on the device",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Song name or artist to play",
                        },
                        "playlist": {
                            "type": "string",
                            "description": "Playlist name (optional)",
                        },
                    },
                    "required": ["query"],
                },
                handler=self._play_music,
            )
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def execute_tool(
        self, name: str, arguments: dict[str, Any], call_id: Optional[str] = None
    ) -> dict[str, Any]:
        start_time = time.time()

        if name not in self._tools:
            logger.error(f"Tool not found: {name}")
            return {
                "success": False,
                "error": f"Tool '{name}' not found",
                "call_id": call_id,
            }

        tool = self._tools[name]
        handler = tool.handler
        if not handler:
            logger.error(f"Tool has no handler: {name}")
            return {
                "success": False,
                "error": f"Tool '{name}' has no handler",
                "call_id": call_id,
            }

        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")

            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: handler(**arguments)
                )

            duration_ms = (time.time() - start_time) * 1000

            tool_call_record = {
                "name": name,
                "arguments": arguments,
                "result": result,
                "success": True,
                "duration_ms": duration_ms,
                "call_id": call_id,
            }
            self._tool_call_history.append(tool_call_record)

            logger.info(f"Tool {name} executed successfully in {duration_ms:.2f}ms")

            return {
                "success": True,
                "result": result,
                "duration_ms": duration_ms,
                "call_id": call_id,
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Tool {name} failed: {error_msg}")

            tool_call_record = {
                "name": name,
                "arguments": arguments,
                "error": error_msg,
                "success": False,
                "duration_ms": duration_ms,
                "call_id": call_id,
            }
            self._tool_call_history.append(tool_call_record)

            return {
                "success": False,
                "error": error_msg,
                "duration_ms": duration_ms,
                "call_id": call_id,
            }

    def get_tool_call_history(self) -> list[dict[str, Any]]:
        return self._tool_call_history.copy()

    def clear_history(self) -> None:
        self._tool_call_history.clear()

    def _get_weather(self, city: str) -> dict[str, Any]:
        weather_data = {
            "Beijing": {"temperature": 15, "condition": "Sunny", "humidity": 45},
            "Shanghai": {"temperature": 18, "condition": "Cloudy", "humidity": 60},
            "Shenzhen": {"temperature": 25, "condition": "Rainy", "humidity": 80},
            "Guangzhou": {"temperature": 26, "condition": "Overcast", "humidity": 75},
        }

        city_lower = city.lower().replace(" ", "")
        for city_name, data in weather_data.items():
            if city_lower in city_name.lower():
                return {
                    "city": city_name,
                    **data,
                    "unit": "celsius",
                }

        return {
            "city": city,
            "temperature": 20,
            "condition": "Unknown",
            "humidity": 50,
            "unit": "celsius",
            "note": "Simulated weather data",
        }

    def _set_volume(self, volume: int) -> dict[str, Any]:
        volume = max(0, min(100, volume))
        logger.info(f"Setting volume to {volume}")
        return {
            "volume": volume,
            "status": "success",
            "message": f"Volume set to {volume}",
        }

    def _play_music(self, query: str, playlist: Optional[str] = None) -> dict[str, Any]:
        logger.info(f"Playing music: {query}, playlist: {playlist}")
        result = {
            "query": query,
            "status": "playing",
            "message": f"Playing '{query}'",
        }
        if playlist:
            result["playlist"] = playlist
        return result
