import asyncio
import json
from typing import Any, Optional
import logging
import aiohttp

from .config import LLMConfig
from .mcp_manager import MCPToolManager

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: LLMConfig, mcp_manager: Optional[MCPToolManager] = None):
        self.config = config
        self.mcp_manager = mcp_manager
        self._session: Optional[aiohttp.ClientSession] = None

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
            print(f"LLM Request:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    raise Exception(f"LLM API error: {response.status}")

                result = await response.json()
                print(
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
        all_messages = messages.copy()

        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        tools = None
        if self.mcp_manager:
            tools = self.mcp_manager.get_openai_tools()

        response = await self.chat_completion(all_messages, tools=tools)

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = message.get("tool_calls", [])

        if tool_calls and self.mcp_manager:
            all_messages.append(message)

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                tool_args_str = function.get("arguments", "{}")
                tool_call_id = tool_call.get("id")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                execute_result = await self.mcp_manager.execute_tool(
                    tool_name, tool_args, tool_call_id
                )

                tool_result_content = json.dumps(
                    execute_result.get(
                        "result", execute_result.get("error", "Unknown error")
                    )
                )

                all_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result_content,
                    }
                )

            final_response = await self.chat_completion(all_messages, tools=tools)
            return final_response

        return response

    def extract_content(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return message.get("content", "")

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        choices = response.get("choices", [])
        if not choices:
            return []

        message = choices[0].get("message", {})
        return message.get("tool_calls", [])
