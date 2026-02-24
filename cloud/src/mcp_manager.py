import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime
import logging
import aiohttp
import requests

from .roleplay_manager import get_roleplay_manager

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
        amap_api_key: Optional[str] = None,
        weather_api_enabled: bool = True,
        roleplay_llm_url: Optional[str] = None,
        roleplay_llm_model: Optional[str] = None,
    ):
        self.server_name = server_name
        self.server_version = server_version
        self.protocol_version = protocol_version
        self.instructions = instructions
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_call_history: list[dict[str, Any]] = []
        self._location_info: dict[str, Any] = {}
        self._amap_api_key = amap_api_key
        self._weather_api_enabled = weather_api_enabled and amap_api_key is not None
        self._roleplay_llm_model = roleplay_llm_model
        self._roleplay_manager = get_roleplay_manager(default_llm_url=roleplay_llm_url)
        self._register_builtin_tools()
        # 异步获取位置信息（不阻塞初始化）
        asyncio.create_task(self._fetch_location_info())

    def _register_builtin_tools(self) -> None:
        self.register_tool(
            ToolDefinition(
                name="get_weather",
                description="Get weather information. IMPORTANT: By default, query current location weather (omit city parameter). Only specify city parameter if user explicitly asks for a different city (like '北京天气', '上海明天如何').",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Target city name (optional). Only specify if user explicitly asks for a different city. Default is current location.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date to query weather for (optional). Can be 'today', 'tomorrow', '后天', '大后天', or specific date like '2024-01-15'. Default is 'today'.",
                        },
                    },
                },
                handler=self._get_weather,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="get_current_time",
                description="Get the current time",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone identifier (e.g., 'Asia/Shanghai', 'UTC'). If not provided, uses system local timezone.",
                        },
                    },
                },
                handler=self._get_current_time,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="get_location",
                description="Get current location information based on IP address, including city, country, region, and timezone",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                handler=self._get_location,
            )
        )

        # 角色扮演相关工具
        self.register_tool(
            ToolDefinition(
                name="enter_roleplay_mode",
                description="Enter roleplay mode. Use this when user explicitly asks to roleplay or pretend to be a specific character (like '扮演至尊宝', '进入角色扮演模式', '拜年模式'). Returns system prompt for the character.",
                parameters={
                    "type": "object",
                    "properties": {
                        "character": {
                            "type": "string",
                            "description": "Character name to roleplay. Available characters: 至尊宝, 紫霞仙子, 牛魔王, 唐僧, 拜年模式",
                            "enum": ["至尊宝", "紫霞仙子", "牛魔王", "唐僧", "拜年模式"],
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                    "required": ["character"],
                },
                handler=self._enter_roleplay_mode,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="exit_roleplay_mode",
                description="Exit roleplay mode. Use this when user explicitly asks to exit roleplay (like '退出角色扮演模式', '停止扮演').",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                },
                handler=self._exit_roleplay_mode,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="get_roleplay_status",
                description="Get current roleplay mode status, including active character and dialogue count.",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                },
                handler=self._get_roleplay_status,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="list_available_characters",
                description="List all available characters for roleplay mode.",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                handler=self._list_available_characters,
            )
        )

        # LLM服务器配置相关工具
        self.register_tool(
            ToolDefinition(
                name="get_roleplay_llm_config",
                description="Get the LLM server configuration for the current roleplay session. Returns the LLM server URL being used.",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                },
                handler=self._get_roleplay_llm_config,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="set_roleplay_llm_url",
                description="Set a custom LLM server URL for the current roleplay session. This overrides the default and character-specific settings.",
                parameters={
                    "type": "object",
                    "properties": {
                        "llm_url": {
                            "type": "string",
                            "description": "LLM server URL (e.g., 'http://localhost:8000/v1/chat/completions', 'https://api.openai.com/v1/chat/completions')",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                    "required": ["llm_url"],
                },
                handler=self._set_roleplay_llm_url,
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

    def get_location_context(self) -> str:
        """获取位置上下文信息，用于系统提示"""
        if self._location_info:
            city = self._location_info.get("city", "Unknown")
            province = self._location_info.get("region", "")
            country = self._location_info.get("country", "")
            if province and province != city:
                return f"Current location: {city}, {province}, {country}"
            return f"Current location: {city}, {country}"
        return ""

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

    def _get_weather_from_api(self, city: str, extensions: str = "base") -> dict[str, Any]:
        """同步调用高德天气 API"""
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "key": self._amap_api_key,
            "city": city,
            "extensions": extensions,
            "output": "JSON",
        }

        # 隐藏 key 的日志
        safe_params = params.copy()
        safe_params["key"] = "***" if self._amap_api_key else None

        logger.info(f"Amap Weather API Request: city={city}, extensions={extensions}")

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            logger.info(f"Amap Weather API Response: {json.dumps(data, ensure_ascii=False)}")

            if data.get("status") == "1":
                return {"success": True, "data": data}
            else:
                return {
                    "success": False,
                    "error": data.get("info", "Unknown error"),
                    "infocode": data.get("infocode"),
                }
        except requests.Timeout:
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            logger.warning(f"Amap Weather API Error: {e}")
            return {"success": False, "error": str(e)}

    def _get_weather(self, city: Optional[str] = None, date: Optional[str] = None) -> dict[str, Any]:
        """获取天气信息"""
        # 城市名称映射表（支持中文和英文）
        city_name_map = {
            # 直辖市
            "beijing": "北京",
            "shanghai": "上海",
            "tianjin": "天津",
            "chongqing": "重庆",
            # 省会及主要城市
            "shenzhen": "深圳",
            "guangzhou": "广州",
            "hangzhou": "杭州",
            "nanjing": "南京",
            "chengdu": "成都",
            "wuhan": "武汉",
            "xian": "西安",
            "fuzhou": "福州",
            # 华北地区
            "shijiazhuang": "石家庄",
            "taiyuan": "太原",
            "hohhot": "呼和浩特",
            # 东北地区
            "shenyang": "沈阳",
            "dalian": "大连",
            "changchun": "长春",
            "harbin": "哈尔滨",
            # 华东地区
            "hefei": "合肥",
            "nanchang": "南昌",
            "jinan": "济南",
            "qingdao": "青岛",
            "ningbo": "宁波",
            "xiamen": "厦门",
            "suzhou": "苏州",
            "wuxi": "无锡",
            # 华中地区
            "zhengzhou": "郑州",
            "changsha": "长沙",
            # 华南地区
            "nanning": "南宁",
            "haikou": "海口",
            "sanya": "三亚",
            # 西南地区
            "kunming": "昆明",
            "guizhou": "贵阳",
            "lhasaa": "拉萨",
            # 西北地区
            "lanzhou": "兰州",
            "yinchuan": "银川",
            "xining": "西宁",
            "urumqi": "乌鲁木齐",
        }

        # 如果没有指定城市，使用获取到的位置信息
        if not city:
            city = self.get_location_city()
            if not city:
                return {
                    "city": "Unknown",
                    "temperature": 20,
                    "condition": "Unknown",
                    "humidity": 50,
                    "unit": "celsius",
                    "note": "Location info not available, please specify a city",
                }

        # 标准化城市名称（转换为中文供高德 API 使用）
        city_lower = city.lower().replace(" ", "")
        city_for_api = city_name_map.get(city_lower, city)

        # 判断是今天还是未来日期
        date_display = date or "今天"
        is_forecast = date in ["tomorrow", "明天", "后天", "大后天", "the day after tomorrow"]

        # 如果启用了真实天气 API
        if self._weather_api_enabled:
            try:
                # 直接调用同步 API
                result = self._get_weather_from_api(
                    city_for_api,
                    "all" if is_forecast else "base"
                )

                if result.get("success"):
                    data = result["data"]

                    # 解析实况天气
                    if "lives" in data and data["lives"]:
                        live = data["lives"][0]
                        return {
                            "city": live.get("city", city),
                            "province": live.get("province"),
                            "date": date_display,
                            "temperature": int(live.get("temperature", 0)),
                            "condition": live.get("weather"),
                            "humidity": int(live.get("humidity", 0)),
                            "wind_direction": live.get("winddirection"),
                            "wind_power": live.get("windpower"),
                            "report_time": live.get("reporttime"),
                            "unit": "celsius",
                            "source": "Amap Weather API",
                        }

                    # 解析预报天气
                    if "forecasts" in data and data["forecasts"]:
                        forecast = data["forecasts"][0]
                        casts = forecast.get("casts", [])

                        # 根据日期选择预报数据
                        target_date = None
                        if date in ["tomorrow", "明天"] and len(casts) > 1:
                            target_date = casts[1]
                            date_display = "明天"
                        elif date in ["后天"] and len(casts) > 2:
                            target_date = casts[2]
                            date_display = "后天"
                        elif date in ["大后天"] and len(casts) > 3:
                            target_date = casts[3]
                            date_display = "大后天"
                        else:
                            target_date = casts[0]

                        if target_date:
                            return {
                                "city": forecast.get("city", city),
                                "date": target_date.get("date"),
                                "date_display": date_display,
                                "week": target_date.get("week"),
                                "day_weather": target_date.get("dayweather"),
                                "night_weather": target_date.get("nightweather"),
                                "day_temp": int(target_date.get("daytemp", 0)),
                                "night_temp": int(target_date.get("nighttemp", 0)),
                                "temperature": int(target_date.get("daytemp", 0)),  # 兼容字段
                                "condition": target_date.get("dayweather"),  # 兼容字段
                                "unit": "celsius",
                                "source": "Amap Weather API (Forecast)",
                            }

                # API 调用失败，使用模拟数据
                logger.warning(f"Amap API failed: {result.get('error')}, using fallback data")

            except Exception as e:
                logger.warning(f"Amap API error: {e}, using fallback data")

        # ========== 以下是模拟数据（API 不可用时使用）==========
        weather_data = {
            "北京": {"temperature": 15, "condition": "晴", "humidity": 45},
            "上海": {"temperature": 18, "condition": "多云", "humidity": 60},
            "深圳": {"temperature": 25, "condition": "雨", "humidity": 80},
            "广州": {"temperature": 26, "condition": "阴", "humidity": 75},
        }

        # 查找匹配的天气数据
        weather_info = None
        for city_name, data in weather_data.items():
            if city_for_api in city_name or city_lower in city_name.lower():
                weather_info = data
                break

        # 如果没有找到，生成基于城市名的模拟天气
        if not weather_info:
            import hashlib
            hash_val = int(hashlib.md5(city_for_api.encode()).hexdigest(), 16)
            temp = 15 + (hash_val % 20)
            conditions = ["晴", "多云", "雨", "阴", "晴转多云"]
            condition = conditions[hash_val % len(conditions)]
            humidity = 40 + (hash_val % 50)
            weather_info = {"temperature": temp, "condition": condition, "humidity": humidity}

        result = {
            "city": city_for_api,
            "date": date_display,
            **weather_info,
            "unit": "celsius",
            "source": "Simulated data (API unavailable)",
        }

        # 为明天/后天的天气添加一些变化
        if is_forecast:
            import hashlib
            date_hash = int(hashlib.md5(f"{city_for_api}{date}".encode()).hexdigest(), 16)
            temp_change = (date_hash % 7) - 3  # -3 到 +3 的温度变化
            result["temperature"] += temp_change
            result["forecast_note"] = f"{date_display}的天气预报（基于模拟数据）"

        return result

    def _get_current_time(self, timezone: Optional[str] = None) -> dict[str, Any]:
        from zoneinfo import ZoneInfo

        if timezone:
            try:
                tz = ZoneInfo(timezone)
                now = datetime.now(tz)
                tz_display = timezone
            except Exception as e:
                logger.warning(f"Invalid timezone '{timezone}': {e}, using local timezone")
                now = datetime.now()
                tz_display = "Local"
        else:
            now = datetime.now()
            tz_display = "Local"

        return {
            "datetime": now.isoformat(),
            "timestamp": int(now.timestamp()),
            "timezone": tz_display,
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "weekday": now.strftime("%A"),
            "date_string": now.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _get_location(self) -> dict[str, Any]:
        """获取当前位置信息"""
        if not self._location_info:
            return {
                "status": "unavailable",
                "message": "Location information is not available yet. Please try again in a moment.",
                "note": "Location is fetched asynchronously during server initialization",
            }

        return {
            "status": "success",
            "ip": self._location_info.get("ip"),
            "country": self._location_info.get("country"),
            "country_code": self._location_info.get("country_code"),
            "region": self._location_info.get("region"),
            "city": self._location_info.get("city"),
            "latitude": self._location_info.get("latitude"),
            "longitude": self._location_info.get("longitude"),
            "timezone": self._location_info.get("timezone"),
        }

    async def _fetch_location_info(self) -> None:
        """获取外网 IP 和位置信息"""
        url = "http://ip-api.com/json/"
        timeout = aiohttp.ClientTimeout(total=5)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            self._location_info = {
                                "ip": data.get("query"),
                                "country": data.get("country"),
                                "country_code": data.get("countryCode"),
                                "region": data.get("regionName"),
                                "city": data.get("city"),
                                "latitude": data.get("lat"),
                                "longitude": data.get("lon"),
                                "timezone": data.get("timezone"),
                            }
                            logger.info(f"Location info fetched: {self._location_info.get('city')}, {self._location_info.get('country')}")
                        else:
                            logger.warning(f"Failed to fetch location info: {data.get('message', 'Unknown error')}")
                    else:
                        logger.warning(f"Failed to fetch location info: HTTP {response.status}")
        except asyncio.TimeoutError:
            logger.warning("Timeout while fetching location info")
        except Exception as e:
            logger.warning(f"Error fetching location info: {e}")

    def get_location_city(self) -> Optional[str]:
        """获取当前位置的城市名称"""
        return self._location_info.get("city")

    def get_location_info(self) -> dict[str, Any]:
        """获取完整的位置信息"""
        return self._location_info.copy()

    # ========== 角色扮演相关方法 ==========

    def _enter_roleplay_mode(
        self,
        character: str,
        session_id: str = "default"
    ) -> dict[str, Any]:
        """进入角色扮演模式"""
        return self._roleplay_manager.enter_roleplay_mode(character, session_id)

    def _exit_roleplay_mode(self, session_id: str = "default") -> dict[str, Any]:
        """退出角色扮演模式"""
        return self._roleplay_manager.exit_roleplay_mode(session_id)

    def _get_roleplay_status(self, session_id: str = "default") -> dict[str, Any]:
        """获取角色扮演状态"""
        return self._roleplay_manager.get_roleplay_status(session_id)

    def _list_available_characters(self) -> dict[str, Any]:
        """列出所有可用角色"""
        characters = self._roleplay_manager.get_available_characters()
        return {
            "success": True,
            "characters": characters,
            "count": len(characters),
        }

    def is_in_roleplay_mode(self, session_id: str = "default") -> bool:
        """检查是否处于角色扮演模式"""
        return self._roleplay_manager.is_in_roleplay_mode(session_id)

    def get_roleplay_system_prompt(self, session_id: str = "default") -> Optional[str]:
        """获取角色扮演的系统提示词"""
        return self._roleplay_manager.get_system_prompt(session_id)

    def add_roleplay_dialogue(
        self,
        session_id: str,
        user_message: str,
        response: str
    ) -> None:
        """添加角色扮演对话记录"""
        self._roleplay_manager.add_dialogue(session_id, user_message, response)

    def get_roleplay_messages(
        self,
        session_id: str,
        user_message: str
    ) -> Optional[list[dict[str, str]]]:
        """获取角色扮演模式的LLM消息列表（OpenAI格式）"""
        return self._roleplay_manager.get_messages_for_completion(session_id, user_message)

    def _get_roleplay_llm_config(self, session_id: str = "default") -> dict[str, Any]:
        """获取角色扮演的LLM配置"""
        return self._roleplay_manager.get_llm_config(session_id)

    def _set_roleplay_llm_url(
        self,
        llm_url: str,
        session_id: str = "default"
    ) -> dict[str, Any]:
        """设置角色扮演会话的LLM服务器地址"""
        self._roleplay_manager.set_session_llm_url(session_id, llm_url)
        return {
            "success": True,
            "message": f"LLM server URL set to: {llm_url}",
            "session_id": session_id,
        }

    def get_roleplay_llm_url(self, session_id: str = "default") -> str:
        """获取角色扮演会话的LLM服务器地址"""
        return self._roleplay_manager.get_llm_server_url(session_id)

    def get_roleplay_llm_model(self) -> Optional[str]:
        """获取角色扮演模式的LLM模型名称"""
        return self._roleplay_llm_model
