import asyncio
import json
from typing import Any, Optional
import logging
import aiohttp
import re

from .config import LLMConfig
from .mcp_manager import MCPToolManager

logger = logging.getLogger(__name__)


# 判断工具是否是客户端工具的标记前缀
CLIENT_TOOL_PREFIX = "client:"


# Emoji 和特殊符号的 Unicode 范围
EMOJI_RANGES = [
    (0x1F600, 0x1F64F),  # 表情符号
    (0x1F300, 0x1F5FF),  # 符号和图标
    (0x1F680, 0x1F6FF),  # 交通和地图
    (0x1F700, 0x1F77F),  # 炼金术符号
    (0x1F780, 0x1F7FF),  # 几何符号
    (0x1F800, 0x1F8FF),  # 补充符号
    (0x1F900, 0x1F9FF),  # 补充符号和图标
    (0x1FA00, 0x1FA6F),  # 扩展符号
    (0x1FA70, 0x1FAFF),  # 符号和图标
    (0x2600, 0x26FF),    # 杂项符号
    (0x2700, 0x27BF),    # 装饰符号
    (0xFE00, 0xFE0F),    # 变体选择器
    (0x1F900, 0x1F9FF),  # 补充符号
    (0x231A, 0x231B),    # 手表/时钟
    (0x23E9, 0x23F3),    # 各种箭头
    (0x23F8, 0x23FA),    # 媒体符号
    (0x25AA, 0x25AB),    # 方形
    (0x25B6, 0x25C0),    # 三角形
    (0x25FB, 0x25FE),    # 方形
    (0x2614, 0x2615),    # 雨伞/热饮
    (0x2648, 0x2653),    # 星座
    (0x267F, 0x267F),    # 轮椅
    (0x2693, 0x2693),    # 锚
    (0x26A1, 0x26A1),    # 高压
    (0x26AA, 0x26AB),    # 圆圈
    (0x26BD, 0x26BE),    # 足球/棒球
    (0x26C4, 0x26C5),    # 雪人/太阳
    (0x26CE, 0x26CE),    # 星座
    (0x26D4, 0x26D4),    # 禁止
    (0x26EA, 0x26EA),    # 教堂
    (0x26F2, 0x26F3),    # 喷泉/高尔夫
    (0x26F5, 0x26F5),    # 帆船
    (0x26FA, 0x26FA),    # 帐篷
    (0x26FD, 0x26FD),    # 燃料泵
    (0x2702, 0x2702),    # 剪刀
    (0x2705, 0x2705),    # 勾
    (0x2708, 0x270D),    # 各种符号
    (0x270F, 0x270F),    # 铅笔
    (0x2712, 0x2712),    # 笔
    (0x2714, 0x2714),    # 勾
    (0x2716, 0x2716),    # 乘
    (0x271D, 0x271D),    # 十字
    (0x2721, 0x2721),    # 星星
    (0x2728, 0x2728),    # 闪光
    (0x2733, 0x2734),    # 星星
    (0x2744, 0x2744),    # 雪花
    (0x2747, 0x2747),    # 闪光
    (0x274C, 0x274C),    # 叉
    (0x274E, 0x274E),    # 叉
    (0x2753, 0x2755),    # 问号
    (0x2757, 0x2757),    # 感叹号
    (0x2763, 0x2764),    # 心形
    (0x2795, 0x2797),    # 加减乘
    (0x27A1, 0x27A1),    # 箭头
    (0x27B0, 0x27B0),    # 循环
    (0x27BF, 0x27BF),    # 双圈
    (0x2934, 0x2935),    # 箭头
    (0x2B05, 0x2B07),    # 箭头
    (0x2B1B, 0x2B1C),    # 方形
    (0x2B50, 0x2B50),    # 星星
    (0x2B55, 0x2B55),    # 圆圈
    (0x3030, 0x3030),    # 波浪
    (0x303D, 0x303D),    # 交替
    (0x3297, 0x3297),    # 圆圈
    (0x3299, 0x3299),    # 圆圈
]


def _remove_emojis_and_special_chars(text: str) -> str:
    """
    移除文本中的表情符号和特殊字符
    """
    if not text:
        return text

    # 移除 emoji
    for start, end in EMOJI_RANGES:
        text = text.encode("utf-32-le").decode("utf-32-le", errors="ignore")

    # 使用正则表达式移除特殊符号
    # 移除各种装饰性符号
    text = re.sub(r"[★☆◆◇○●■□△▽▲▼◎◢◣◤◥♠♥♦♣♤♡♧♢♀♂☀☁☂☃☄★☆]", "", text)
    # 移除箭头类符号
    text = re.sub(r"[←↑→↓↔↕↖↗↘↙⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏⇐⇑⇒⇓⇔⇕⇖⇗⇘⇙]", "", text)
    # 移除其他特殊符号
    text = re.sub(r"[©®™℠℞℻ℼℽℾℿ⅀⅁⅂⅃⅄ⅅⅆⅇⅈⅉ⅊⅋⅌⅍ⅎ⅏]", "", text)

    # 移除零宽字符
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # 再次使用 unicode 范围清理 emoji
    cleaned_chars = []
    for char in text:
        code = ord(char)
        is_emoji = any(start <= code <= end for start, end in EMOJI_RANGES)
        if not is_emoji:
            cleaned_chars.append(char)
    text = "".join(cleaned_chars)

    return text


def clean_response_content(text: str) -> str:
    """
    清理 LLM 响应内容，移除不适合语音播报的内容
    """
    if not text:
        return text

    # 移除表情符号和特殊字符
    text = _remove_emojis_and_special_chars(text)

    # 清理多余的空白字符
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # 多个空行压缩为两个
    text = re.sub(r"[ \t]+", " ", text)  # 多个空格压缩为一个
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # 行首行尾空白
    text = re.sub("\n<tool_call>", "", text)

    return text


class LLMClient:
    def __init__(self, config: LLMConfig, mcp_manager: Optional[MCPToolManager] = None):
        self.config = config
        self.mcp_manager = mcp_manager
        self._session: Optional[aiohttp.ClientSession] = None
        self._client_tools: Optional[Any] = None  # ClientToolManager instance

    def set_client_tools(self, client_tools_manager: Any) -> None:
        """设置客户端工具管理器"""
        self._client_tools = client_tools_manager

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            session = await self._get_session()
            logger.debug(f"LLM Request:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    raise Exception(f"LLM API error: {response.status}")

                result = await response.json()
                logger.info(
                    f"LLM Response:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
                )
                return result

        except asyncio.TimeoutError:
            logger.error("LLM request timeout")
            raise Exception("LLM request timeout")
        except aiohttp.ClientError as e:
            logger.error(f"LLM request failed: {e}")
            raise Exception(f"LLM request failed: {e}")

    async def process_with_tools(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        处理带工具的消息（只处理服务端工具）

        客户端工具需要由 server.py 处理回调流程
        """
        all_messages = messages.copy()

        # 使用配置的系统提示词，如果没有传入则使用默认的
        prompt = system_prompt or self.config.system_prompt

        # 添加位置上下文到系统提示
        if self.mcp_manager:
            location_context = self.mcp_manager.get_location_context()
            if location_context:
                logger.info(f"Adding location context to system prompt: {location_context}")
                if prompt:
                    prompt = f"{prompt}\n\n# Location Context\n{location_context}\nWhen user asks about weather without specifying a city, use the current location above."
                else:
                    prompt = f"# Location Context\n{location_context}\nWhen user asks about weather without specifying a city, use the current location above."

        if prompt:
            all_messages.insert(0, {"role": "system", "content": prompt})

        # 收集所有工具（服务端 MCP + 客户端工具）
        tools = []
        if self.mcp_manager:
            tools.extend(self.mcp_manager.get_openai_tools())
        if self._client_tools:
            tools.extend(self._client_tools.get_openai_tools())

        tools_param = tools if tools else None

        logger.info(f"process_with_tools: _client_tools exists: {self._client_tools is not None}")
        if self._client_tools:
            logger.info(f"  Client tool names: {self.get_client_tool_names()}")

        response = await self.chat_completion(all_messages, tools=tools_param)

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = message.get("tool_calls", [])
        logger.info(f"process_with_tools: tool_calls count: {len(tool_calls)}")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                logger.info(f"  Tool call: {func.get('name')} with args: {func.get('arguments')}")

        if tool_calls:
            all_messages.append(message)

            # Track which tools are client tools
            client_tools_found = []

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                tool_args_str = function.get("arguments", "{}")
                tool_call_id = tool_call.get("id")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                # 判断是服务端工具还是客户端工具
                is_client_tool = (
                    self._client_tools and
                    self._client_tools.has_tool(tool_name)
                )

                if is_client_tool:
                    client_tools_found.append(tool_name)

                tool_result_content = ""
                if is_client_tool:
                    # 客户端工具：返回特殊标记，由 server 处理
                    tool_result_content = json.dumps({
                        "_client_tool": True,
                        "tool_name": tool_name,
                    })
                else:
                    # 服务端工具：直接执行
                    if self.mcp_manager:
                        execute_result = await self.mcp_manager.execute_tool(
                            tool_name, tool_args, tool_call_id
                        )
                        logger.info(f"Server tool {tool_name} result: {execute_result.get('success')}")
                        tool_result_content = json.dumps(
                            execute_result.get(
                                "result", execute_result.get("error", "Unknown error")
                            )
                        )
                    else:
                        tool_result_content = json.dumps({"error": "Tool not found"})

                all_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result_content,
                    }
                )

            # 如果有客户端工具调用，不再次调用 LLM，返回当前状态
            # 让 server 处理客户端工具后再次调用
            if client_tools_found:
                logger.info(f"Found {len(client_tools_found)} client tools: {client_tools_found}")
                # 返回包含 tool_calls 的响应，让 server 知道有哪些工具调用
                return response
            else:
                # 只有服务端工具，正常完成流程
                logger.info("All tools are server tools, calling LLM for final response...")
                final_response = await self.chat_completion(all_messages, tools=tools_param)
                logger.info(f"Final LLM response received, content: {final_response.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}")
                return final_response

        return response

    async def continue_with_client_tool_results(
        self,
        messages: list[dict[str, str]],
        tool_continuation: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        在客户端工具执行完成后继续处理

        Args:
            messages: 原始消息列表（包含user消息和历史）
            tool_continuation: 工具调用后的消息序列（assistant消息 + tool结果）

        Returns:
            LLM 最终响应
        """
        all_messages = messages.copy()

        # 添加系统提示和位置上下文
        prompt = self.config.system_prompt

        # 添加位置上下文到系统提示
        if self.mcp_manager:
            location_context = self.mcp_manager.get_location_context()
            if location_context:
                if prompt:
                    prompt = f"{prompt}\n\n# Location Context\n{location_context}\nWhen user asks about weather without specifying a city, use the current location above."
                else:
                    prompt = f"# Location Context\n{location_context}\nWhen user asks about weather without specifying a city, use the current location above."

        if prompt:
            all_messages.insert(0, {"role": "system", "content": prompt})

        # 收集工具定义
        tools = []
        if self.mcp_manager:
            tools.extend(self.mcp_manager.get_openai_tools())
        if self._client_tools:
            tools.extend(self._client_tools.get_openai_tools())

        tools_param = tools if tools else None

        # 添加工具调用后续消息（assistant + tool results）
        for result in tool_continuation:
            all_messages.append(result)

        logger.info(f"Final messages to LLM: {len(all_messages)} messages")

        # 获取最终响应
        final_response = await self.chat_completion(all_messages, tools=tools_param)
        return final_response

    def get_client_tool_names(self) -> list[str]:
        """获取所有客户端工具名称"""
        if self._client_tools:
            return list(self._client_tools._tools.keys())
        return []

    def extract_content(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        # 清理响应内容，移除特殊符号和 markdown 格式
        return clean_response_content(content)

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        choices = response.get("choices", [])
        if not choices:
            return []

        message = choices[0].get("message", {})
        return message.get("tool_calls", [])
