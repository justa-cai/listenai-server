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
            logger.info(f"LLM Request:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
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

    async def chat_completion_roleplay(
        self,
        messages: list[dict[str, str]],
        llm_url: str,
        model: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        角色扮演模式的 LLM 调用（使用自定义 LLM 服务器）

        Args:
            messages: 消息列表
            llm_url: LLM 服务器 URL
            model: 模型名称（可选，默认使用配置的模型）
            temperature: 温度参数
            max_tokens: 最大 token 数
            tools: 可选的工具列表（角色扮演模式下只提供退出/切换角色的工具）

        Returns:
            LLM 响应结果
        """
        url = f"{llm_url.rstrip('/')}/chat/completions"

        headers = {"Content-Type": "application/json"}
        # 尝试从环境变量或配置获取 API key
        api_key = self.config.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 只在角色扮演模式下添加特定的工具
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            # 创建新的 session 以使用不同的超时设置
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug(f"Roleplay LLM Request: {url}\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Roleplay LLM API error: {response.status} - {error_text}")
                        raise Exception(f"Roleplay LLM API error: {response.status}")

                    result = await response.json()
                    logger.info(
                        f"Roleplay LLM Response:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
                    )
                    return result

        except asyncio.TimeoutError:
            logger.error("Roleplay LLM request timeout")
            raise Exception("Roleplay LLM request timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Roleplay LLM request failed: {e}")
            raise Exception(f"Roleplay LLM request failed: {e}")

    async def process_with_tools(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        处理带工具的消息（只处理服务端工具）

        客户端工具需要由 server.py 处理回调流程

        注意：在角色扮演模式下，会自动切换到纯对话模式，不使用任何工具。
        """
        all_messages = messages.copy()

        # 检查是否处于角色扮演模式
        is_roleplay = (
            self.mcp_manager and
            self.mcp_manager.is_in_roleplay_mode(session_id)
        )

        if is_roleplay:
            # 角色扮演模式：只提供退出角色扮演的工具
            logger.info(f"Roleplay mode detected for session {session_id}, using roleplay LLM with exit tool")
            llm_url = self.mcp_manager.get_roleplay_llm_url(session_id)

            # 获取角色扮演消息（包含上下文）
            roleplay_messages = self.mcp_manager.get_roleplay_messages(session_id, "")
            if roleplay_messages:
                # 更新最后一条用户消息
                if messages and messages[-1].get("role") == "user":
                    roleplay_messages[-1]["content"] = messages[-1]["content"]
                    all_messages = roleplay_messages

            # 只提供角色扮演控制工具（退出和切换角色）
            roleplay_tools = []
            if self.mcp_manager:
                # 获取所有工具，但只保留角色扮演相关的工具
                all_tools = self.mcp_manager.get_openai_tools()
                allowed_tools = ["exit_roleplay_mode", "enter_roleplay_mode"]
                for tool in all_tools:
                    tool_name = tool.get("function", {}).get("name")
                    if tool_name in allowed_tools:
                        roleplay_tools.append(tool)

            # 获取角色扮演模型配置
            roleplay_model = self.mcp_manager.get_roleplay_llm_model()

            response = await self.chat_completion_roleplay(
                messages=all_messages,
                llm_url=llm_url,
                model=roleplay_model,
                tools=roleplay_tools if roleplay_tools else None,
            )

            # 检查是否有工具调用
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                logger.info(f"Roleplay mode: detected {len(tool_calls)} tool calls")
                all_messages.append(message)

                exited_roleplay = False
                switched_character = False

                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_name = function.get("name", "")
                    tool_args_str = function.get("arguments", "{}")
                    tool_call_id = tool_call.get("id")

                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # 处理 exit_roleplay_mode 工具
                    if tool_name == "exit_roleplay_mode" and self.mcp_manager:
                        logger.info(f"Executing exit_roleplay_mode tool")
                        # 注入正确的 session_id
                        tool_args["session_id"] = session_id
                        execute_result = await self.mcp_manager.execute_tool(
                            tool_name, tool_args, tool_call_id
                        )
                        tool_result_content = json.dumps(
                            execute_result.get(
                                "result", execute_result.get("error", "Unknown error")
                            )
                        )

                        all_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_result_content,
                        })
                        exited_roleplay = True

                    # 处理 enter_roleplay_mode 工具（切换角色）
                    elif tool_name == "enter_roleplay_mode" and self.mcp_manager:
                        character = tool_args.get("character", "")
                        logger.info(f"Executing enter_roleplay_mode tool to switch to character: {character}")
                        # 注入正确的 session_id
                        tool_args["session_id"] = session_id
                        execute_result = await self.mcp_manager.execute_tool(
                            tool_name, tool_args, tool_call_id
                        )
                        tool_result_content = json.dumps(
                            execute_result.get(
                                "result", execute_result.get("error", "Unknown error")
                            )
                        )

                        all_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_result_content,
                        })
                        switched_character = True
                    else:
                        logger.warning(f"Roleplay mode: ignoring tool call '{tool_name}' (only roleplay control tools allowed)")
                        # 忽略其他工具调用
                        all_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": json.dumps({"error": f"Tool '{tool_name}' is not available in roleplay mode"}),
                        })

                # 如果退出了角色扮演，使用普通 LLM 获取最终回复
                if exited_roleplay:
                    logger.info("Exited roleplay mode, getting final response with normal LLM")
                    return await self.chat_completion(all_messages, tools=None)

                # 如果切换了角色，使用新角色继续对话
                if switched_character:
                    logger.info("Switched character in roleplay mode, continuing with new character")
                    new_llm_url = self.mcp_manager.get_roleplay_llm_url(session_id)
                    roleplay_model = self.mcp_manager.get_roleplay_llm_model()
                    new_roleplay_messages = self.mcp_manager.get_roleplay_messages(session_id, "")
                    if new_roleplay_messages:
                        return await self.chat_completion_roleplay(
                            messages=new_roleplay_messages,
                            llm_url=new_llm_url,
                            model=roleplay_model,
                        )

                return response

            return response

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
                        # 对于角色扮演相关的工具，注入正确的 session_id
                        if tool_name in ["enter_roleplay_mode", "exit_roleplay_mode", "get_roleplay_status"]:
                            tool_args["session_id"] = session_id
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
        tool_continuation: list[dict[str, Any]],
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        在客户端工具执行完成后继续处理

        Args:
            messages: 原始消息列表（包含user消息和历史）
            tool_continuation: 工具调用后的消息序列（assistant消息 + tool结果）
            session_id: 会话ID（用于检测角色扮演模式）

        Returns:
            LLM 最终响应
        """
        # 检查是否处于角色扮演模式（理论上不应该进入这里，因为角色扮演不使用工具）
        is_roleplay = (
            self.mcp_manager and
            self.mcp_manager.is_in_roleplay_mode(session_id)
        )

        if is_roleplay:
            logger.warning(f"continue_with_client_tool_results called in roleplay mode for session {session_id} - should not happen!")
            # 角色扮演模式：忽略工具结果，直接使用角色扮演 LLM
            llm_url = self.mcp_manager.get_roleplay_llm_url(session_id)
            roleplay_model = self.mcp_manager.get_roleplay_llm_model()
            roleplay_messages = self.mcp_manager.get_roleplay_messages(session_id, "")
            if roleplay_messages:
                all_messages = roleplay_messages
            else:
                all_messages = messages.copy()
            return await self.chat_completion_roleplay(messages=all_messages, llm_url=llm_url, model=roleplay_model)
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

    async def chat_with_roleplay_detection(
        self,
        messages: list[dict[str, str]],
        session_id: str = "default",
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        智能聊天：自动检测角色扮演模式并使用相应的 LLM 服务器

        注意：在角色扮演模式下，tools 参数会被忽略（纯对话模式）。

        Args:
            messages: 消息列表
            session_id: 会话ID
            tools: 工具列表（可选，角色扮演模式下忽略）

        Returns:
            LLM 响应结果
        """
        # 检查是否处于角色扮演模式
        is_roleplay = (
            self.mcp_manager and
            self.mcp_manager.is_in_roleplay_mode(session_id)
        )

        if is_roleplay:
            # 使用角色扮演模式的 LLM 服务器
            llm_url = self.mcp_manager.get_roleplay_llm_url(session_id)
            logger.info(f"Using roleplay LLM: {llm_url} (tools disabled in roleplay mode)")

            # 如果需要使用角色扮演的消息格式（包含上下文）
            roleplay_messages = self.mcp_manager.get_roleplay_messages(session_id, "")
            if roleplay_messages:
                # 使用角色扮演管理器生成的消息（包含完整上下文）
                # 但是需要用当前的消息更新最后一条用户消息
                if messages and messages[-1].get("role") == "user":
                    roleplay_messages[-1]["content"] = messages[-1]["content"]
                    messages = roleplay_messages

            return await self.chat_completion_roleplay(
                messages=messages,
                llm_url=llm_url,
            )
        else:
            # 使用默认 LLM 服务器
            return await self.chat_completion(messages, tools=tools)
