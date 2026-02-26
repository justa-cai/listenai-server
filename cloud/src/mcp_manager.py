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
from .girlfriend_manager import get_girlfriend_manager

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
        self._girlfriend_manager = get_girlfriend_manager()
        # Session 语言设置存储
        self._session_languages: dict[str, str] = {}  # session_id -> language_code
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

        # 语言设置相关工具
        self.register_tool(
            ToolDefinition(
                name="set_response_language",
                description="Set the language for LLM responses. Use this when user asks to change response language (like '用英语回答', '请说日语', 'Answer in English'). Supports 10 major languages.",
                parameters={
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "description": "Language code for response",
                            "enum": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, auto-filled)",
                        },
                    },
                    "required": ["language"],
                },
                handler=self._set_response_language,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="get_response_language",
                description="Get the current response language setting.",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, auto-filled)",
                        },
                    },
                },
                handler=self._get_response_language,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="list_supported_languages",
                description="List all supported languages for response.",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                handler=self._list_supported_languages,
            )
        )

        # 女友模式相关工具
        self.register_tool(
            ToolDefinition(
                name="enter_girlfriend_mode",
                description="Enter girlfriend mode. Use this when user explicitly asks to enter girlfriend mode (like '进入女友模式', '韩国女友', '日本女友', '美国女友', '法国女友'). The girlfriend will respond in her native language (Korean, Japanese, English, or French).",
                parameters={
                    "type": "object",
                    "properties": {
                        "girlfriend_type": {
                            "type": "string",
                            "description": "Girlfriend type to enter. Available types: korean_girlfriend (김민지), japanese_girlfriend (さくら), american_girlfriend (Emily), french_girlfriend (Sophie)",
                            "enum": ["korean_girlfriend", "japanese_girlfriend", "american_girlfriend", "french_girlfriend"],
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                    "required": ["girlfriend_type"],
                },
                handler=self._enter_girlfriend_mode,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="exit_girlfriend_mode",
                description="Exit girlfriend mode. Use this when user explicitly asks to exit girlfriend mode (like '退出女友模式', '停止女友模式').",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                },
                handler=self._exit_girlfriend_mode,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="get_girlfriend_status",
                description="Get current girlfriend mode status, including active girlfriend and dialogue count.",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier (optional, defaults to 'default')",
                        },
                    },
                },
                handler=self._get_girlfriend_status,
            )
        )

        self.register_tool(
            ToolDefinition(
                name="list_available_girlfriends",
                description="List all available girlfriends for girlfriend mode.",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                handler=self._list_available_girlfriends,
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
        # 模式互斥：进入角色扮演模式时，如果处于女友模式，先退出女友模式
        if self._girlfriend_manager.is_in_girlfriend_mode(session_id):
            logger.info(f"Session {session_id}: exiting girlfriend mode before entering roleplay mode")
            self._girlfriend_manager.exit_girlfriend_mode(session_id)

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

    # ========== 语言设置相关方法 ==========

    # 支持的语言列表
    SUPPORTED_LANGUAGES = {
        "zh": "中文",
        "en": "English",
        "ja": "日本語",
        "ko": "한국어",
        "de": "Deutsch",
        "fr": "Français",
        "ru": "Русский",
        "pt": "Português",
        "es": "Español",
        "it": "Italiano",
    }

    def _set_response_language(self, language: str, session_id: str = "default") -> dict[str, Any]:
        """
        设置回复语言

        Args:
            language: 语言代码
            session_id: 会话ID
        """
        logger.info(f"_set_response_language called with language={language}, session_id={session_id}")
        logger.info(f"Current _session_languages keys: {list(self._session_languages.keys())}")

        if language not in self.SUPPORTED_LANGUAGES:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "supported_languages": list(self.SUPPORTED_LANGUAGES.keys()),
                "language_names": self.SUPPORTED_LANGUAGES,
            }

        # 存储语言设置
        self._session_languages[session_id] = language
        logger.info(f"Session {session_id}: Language set to {language} ({self.SUPPORTED_LANGUAGES[language]})")
        logger.info(f"Updated _session_languages: {self._session_languages}")

        return {
            "success": True,
            "language": language,
            "language_name": self.SUPPORTED_LANGUAGES[language],
            "message": f"Response language set to {self.SUPPORTED_LANGUAGES[language]}",
        }

    def get_session_language(self, session_id: str = "default") -> Optional[str]:
        """
        获取会话的回复语言设置

        Args:
            session_id: 会话ID

        Returns:
            语言代码，如果未设置则返回 None
        """
        language = self._session_languages.get(session_id)
        logger.debug(f"get_session_language({session_id}) = {language}, all keys: {list(self._session_languages.keys())}")
        return language

    def _get_response_language(self, session_id: str = "default") -> dict[str, Any]:
        """
        获取当前回复语言设置
        """
        language = self._session_languages.get(session_id, "zh")
        language_name = self.SUPPORTED_LANGUAGES.get(language, "中文")

        return {
            "success": True,
            "language": language,
            "language_name": language_name,
            "supported_languages": list(self.SUPPORTED_LANGUAGES.keys()),
            "language_names": self.SUPPORTED_LANGUAGES,
        }

    def _list_supported_languages(self) -> dict[str, Any]:
        """列出所有支持的语言"""
        return {
            "success": True,
            "languages": [
                {"code": code, "name": name}
                for code, name in self.SUPPORTED_LANGUAGES.items()
            ],
            "count": len(self.SUPPORTED_LANGUAGES),
        }

    @staticmethod
    def get_language_instruction(language: str) -> str:
        """
        获取语言指令用于系统提示

        Args:
            language: 语言代码 (zh, en, ja, ko, de, fr, ru, pt, es, it)

        Returns:
            语言指令字符串（包含系统提示替换）
        """
        # 多语言系统提示
        system_prompts = {
            "zh": "你是一个专业的语音助手助手。请遵循以下规范：\n1. 回复简洁明了，直接回答用户问题\n2. 禁止使用任何表情符号、特殊符号（如 emoji、★、◆、● 等）\n3. 使用纯文本格式，避免使用 markdown 格式（如 **加粗**、*斜体*、`代码` 等）\n4. 回复内容适合语音播报，使用自然口语化的表达\n5. 不要重复用户的问题，直接给出答案或建议",
            "en": "You are a professional voice assistant. Please follow these guidelines:\n1. Respond concisely and directly to user questions\n2. Do NOT use any emojis or special symbols (emoji, ★, ◆, ●, etc.)\n3. Use plain text format, avoid markdown format (like **bold**, *italic*, `code`, etc.)\n4. Response content should be suitable for voice broadcasting, using natural conversational expression\n5. Do not repeat the user's question, directly provide the answer or suggestion",
            "ja": "あなたはプロの音声アシスタントです。以下のガイドラインに従ってください：\n1. 簡潔明瞭に、ユーザーの質問に直接答えてください\n2. 絵文字や特殊記号（emoji、★、◆、● など）を使用しないでください\n3. プレーンテキスト形式を使用し、markdown形式（**太字**、*斜体*、`コード` など）を避けてください\n4. 回答内容は音声読み上げに適した、自然な口語表現を使用してください\n5. ユーザーの質問を繰り返さず、直接回答または提案を提供してください",
            "ko": "당신은 전문 음성 어시스턴트입니다. 다음 지침을 따르십시오：\n1. 간결명료하게 사용자 질문에 직접 답변하십시오\n2. 이모티콘이나 특수 기호(emoji, ★, ◆, ● 등)를 사용하지 마십시오\n3. 일반 텍스트 형식을 사용하고, markdown 형식(**볼드체*, *기울임*, `코드` 등)을 피하십시오\n4. 답변 내용은 음성 방송에 적합한 자연스러운 구어체를 사용하십시오\n5. 사용자의 질문을 반복하지 말고 직접 답변이나 제안을 제공하십시오",
            "de": "Sie sind ein professioneller Sprachassistent. Bitte befolgen Sie diese Richtlinien：\n1. Antworten Sie prägnant und direkt auf Benutzerfragen\n2. Verwenden Sie keine Emojis oder Sonderzeichen (emoji, ★, ◆, ●, etc.)\n3. Verwenden Sie reines Textformat, vermeiden Sie Markdown-Format (wie **Fett**, *Kursiv*, `Code`, etc.)\n4. Die Antwort sollte für Sprachausgabe geeignet sein und natürliche Umgangssprache verwenden\n5. Wiederholen Sie nicht die Frage des Benutzers, sondern geben Sie direkt die Antwort oder den Vorschlag",
            "fr": "Vous êtes un assistant vocal professionnel. Veuillez suivre ces directives：\n1. Répondez de manière concise et directe aux questions des utilisateurs\n2. N'utilisez PAS d'émojis ou de symboles spéciaux (emoji, ★, ◆, ●, etc.)\n3. Utilisez du texte brut, évitez le format markdown (comme **gras**, *italique*, `code`, etc.)\n4. Le contenu de la réponse doit être adapté à la synthèse vocale, en utilisant une expression conversationnelle naturelle\n5. Ne répétez pas la question de l'utilisateur, fournissez plutôt directement la réponse ou la suggestion",
            "ru": "Вы профессиональный голосовой помощник. Пожалуйста, следуйте этим рекомендациям：\n1. Отвечайте кратко и прямо на вопросы пользователей\n2. НЕ используйте эмодзи или специальные символы (emoji, ★, ◆, ● и т.д.)\n3. Используйте простой текст, избегайте формата markdown (например, **жирный**, *курсив*, `код` и т.д.)\n4. Содержимое ответа должно быть подходящим для голосового воспроизведения, используйте естественную разговорную речь\n5. Не повторяйте вопрос пользователя, а сразу предоставьте ответ или предложение",
            "pt": "Você é um assistente de voz profissional. Por favor, siga estas diretrizes：\n1. Responda de forma concisa e direta às perguntas dos usuários\n2. NÃO use emojis ou símbolos especiais (emoji, ★, ◆, ●, etc.)\n3. Use formato de texto simples, evite formato markdown (como **negrito**, *itálico*, `código`, etc.)\n4. O conteúdo da resposta deve ser adequado para transmissão de voz, usando expressão conversacional natural\n5. Não repita a pergunta do usuário, forneça diretamente a resposta ou sugestão",
            "es": "Eres un asistente de voz profesional. Por favor, sigue estas directrices：\n1. Responde de manera concisa y directa a las preguntas de los usuarios\n2. NO uses emojis o símbolos especiales (emoji, ★, ◆, ●, etc.)\n3. Usa formato de texto plano, evita formato markdown (como **negrita**, *cursiva*, `código`, etc.)\n4. El contenido de la respuesta debe ser adecuado para transmisión de voz, usando expresión conversacional natural\n5. No repitas la pregunta del usuario, proporciona directamente la respuesta o sugerencia",
            "it": "Sei un assistente vocale professionale. Si prega di seguire queste linee guida：\n1. Rispondi in modo conciso e diretto alle domande degli utenti\n2. NON usare emoji o simboli speciali (emoji, ★, ◆, ●, ecc.)\n3. Usa formato di testo semplice, evita il formato markdown (come **grassetto**, *corsivo*, `codice`, ecc.)\n4. Il contenuto della risposta deve essere adatto alla sintesi vocale, usando espressione conversazionale naturale\n5. Non ripetere la domanda dell'utente, ma fornisci direttamente la risposta o il suggerimento",
        }

        base_prompt = system_prompts.get(language, system_prompts["zh"])

        instructions = {
            "zh": f"""

{base_prompt}

====================================
【重要】回复语言：中文
====================================
你必须用中文回复所有内容。这是最高优先级的指令，必须严格遵守。
====================================
""",
            "en": f"""

{base_prompt}

====================================
[IMPORTANT] Language: ENGLISH
====================================
You MUST respond in ENGLISH to everything.
This is the HIGHEST PRIORITY instruction and must be strictly followed.
====================================
""",
            "ja": f"""

{base_prompt}

====================================
【重要】言語：日本語
====================================
あなたは日本語で回答しなければなりません。
これは最高優先順位の指示であり、厳守してください。
====================================
""",
            "ko": f"""

{base_prompt}

====================================
[중요] 언어: 한국어
====================================
반드시 한국어로 답변해야 합니다.
이는 최고 우선순위 지시이며 반드시 준수해야 합니다.
====================================
""",
            "de": f"""

{base_prompt}

====================================
[WICHTIG] Sprache: DEUTSCH
====================================
Sie MÜSSEN auf Deutsch antworten.
Dies ist die HÖCHSTE PRIORITÄT und muss strikt befolgt werden.
====================================
""",
            "fr": f"""

{base_prompt}

====================================
[IMPORTANT] Langue: FRANÇAIS
====================================
Vous DEVEZ répondre en français.
Ceci est l'instruction de PRIORITÉ MAXIMALE et doit être strictement respectée.
====================================
""",
            "ru": f"""

{base_prompt}

====================================
[ВАЖНО] Язык: РУССКИЙ
====================================
Вы ДОЛЖНЫ отвечать по-русски.
Это инструкция НАИВЫСШЕГО ПРИОРИТЕТА и должна строго соблюдаться.
====================================
""",
            "pt": f"""

{base_prompt}

====================================
[IMPORTANTE] Idioma: PORTUGUÊS
====================================
Você DEVE responder em português.
Esta é a instrução de PRIORIDADE MÁXIMA e deve ser estritamente seguida.
====================================
""",
            "es": f"""

{base_prompt}

====================================
[IMPORTANTE] Idioma: ESPAÑOL
====================================
DEBES responder en español.
Esta es la instrucción de PRIORIDAD MÁXIMA y debe seguirse estrictamente.
====================================
""",
            "it": f"""

{base_prompt}

====================================
[IMPORTANTE] Lingua: ITALIANO
====================================
DEVI rispondere in italiano.
Questa è l'istruzione di PRIORITÀ MASSIMA e deve essere seguita rigorosamente.
====================================
""",
        }

        return instructions.get(language, instructions["zh"])

    # ========== 女友模式相关方法 ==========

    def _enter_girlfriend_mode(
        self,
        girlfriend_type: str,
        session_id: str = "default"
    ) -> dict[str, Any]:
        """进入女友模式"""
        # 模式互斥：进入女友模式时，如果处于角色扮演模式，先退出角色扮演模式
        if self._roleplay_manager.is_in_roleplay_mode(session_id):
            logger.info(f"Session {session_id}: exiting roleplay mode before entering girlfriend mode")
            self._roleplay_manager.exit_roleplay_mode(session_id)

        return self._girlfriend_manager.enter_girlfriend_mode(girlfriend_type, session_id)

    def _exit_girlfriend_mode(self, session_id: str = "default") -> dict[str, Any]:
        """退出女友模式"""
        return self._girlfriend_manager.exit_girlfriend_mode(session_id)

    def _get_girlfriend_status(self, session_id: str = "default") -> dict[str, Any]:
        """获取女友模式状态"""
        return self._girlfriend_manager.get_girlfriend_status(session_id)

    def _list_available_girlfriends(self) -> dict[str, Any]:
        """列出所有可用女友"""
        girlfriends = self._girlfriend_manager.get_available_girlfriends()
        return {
            "success": True,
            "girlfriends": girlfriends,
            "count": len(girlfriends),
        }

    def is_in_girlfriend_mode(self, session_id: str = "default") -> bool:
        """检查是否处于女友模式"""
        return self._girlfriend_manager.is_in_girlfriend_mode(session_id)

    def get_girlfriend_system_prompt(self, session_id: str = "default") -> Optional[str]:
        """获取女友模式的系统提示词"""
        return self._girlfriend_manager.get_system_prompt(session_id)

    def add_girlfriend_dialogue(
        self,
        session_id: str,
        user_message: str,
        response: str
    ) -> None:
        """添加女友模式对话记录"""
        self._girlfriend_manager.add_dialogue(session_id, user_message, response)

    def get_girlfriend_messages(
        self,
        session_id: str,
        user_message: str
    ) -> Optional[list[dict[str, str]]]:
        """获取女友模式的LLM消息列表（OpenAI格式）"""
        return self._girlfriend_manager.get_messages_for_completion(session_id, user_message)

    def get_girlfriend_character_language(self, session_id: str = "default") -> Optional[str]:
        """获取女友角色的语言代码"""
        return self._girlfriend_manager.get_character_language(session_id)
