import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional
import logging
import websockets
from aiohttp import web

from .config import Config, setup_logging
from .protocol import (
    ClientMessageType,
    ServerMessageType,
    ErrorCode,
    parse_client_message,
    LLMResponseMessage,
    PongMessage,
    ToolCallMessage,
    ToolCallbackMessage,
    ToolsRegisteredMessage,
    create_error_message,
    create_status_message,
)
from .session import SessionManager, Session
from .llm_client import LLMClient
from .mcp_manager import MCPToolManager
from .client_tools import ClientToolManager

logger = logging.getLogger(__name__)


class ClientConnection:
    def __init__(
        self,
        websocket: Any,
        session: Session,
        config: Config,
        llm_client: LLMClient,
        mcp_manager: Optional[MCPToolManager],
        client_tool_manager: Optional[ClientToolManager] = None,
    ):
        self.websocket = websocket
        self.session = session
        self.config = config
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.client_tool_manager = client_tool_manager
        self.is_processing = False
        self._lock = asyncio.Lock()
        # Store pending LLM continuation state (for client tools)
        self._pending_llm_state: Optional[dict[str, Any]] = None

    async def send_json(self, message: dict[str, Any]) -> None:
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")


class CloudServer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.logger = setup_logging(self.config)
        self.session_manager = SessionManager(
            timeout_seconds=self.config.server.session_timeout
        )
        self.mcp_manager = (
            MCPToolManager(
                server_name=self.config.mcp.server_name,
                server_version=self.config.mcp.server_version,
                protocol_version=self.config.mcp.protocol_version,
                instructions=self.config.mcp.instructions,
                amap_api_key=self.config.mcp.amap_api_key,
                weather_api_enabled=self.config.mcp.weather_api_enabled,
            )
            if self.config.mcp.enabled
            else None
        )

        self._connections: dict[str, ClientConnection] = {}
        self._connection_count = 0
        self._running = False

    def _create_client_tool_manager(self) -> Optional[ClientToolManager]:
        """创建客户端工具管理器"""
        if self.config.client_tools.enabled:
            return ClientToolManager(
                max_tools=self.config.client_tools.max_tools,
                tool_timeout=self.config.client_tools.tool_timeout,
                result_queue_size=self.config.client_tools.result_queue_size,
            )
        return None

    async def start(self) -> None:
        self._running = True

        async with websockets.serve(
            self._handle_connection,
            self.config.server.host,
            self.config.server.port,
            ping_interval=self.config.server.ping_interval,
            ping_timeout=self.config.server.ping_timeout,
            max_size=1024 * 1024,
        ):
            logger.info(
                f"Cloud server (LLM Gateway) started on {self.config.server.host}:{self.config.server.port}"
            )

            asyncio.create_task(self._cleanup_task())

            await asyncio.Future()

    async def _cleanup_task(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            cleaned = self.session_manager.cleanup_expired()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")

    async def _handle_connection(self, websocket: Any) -> None:
        self._connection_count += 1
        if len(self._connections) >= self.config.server.max_connections:
            logger.warning("Max connections reached, rejecting new connection")
            await websocket.close(1013, "Max connections reached")
            return

        connection_id = f"conn_{id(websocket)}"
        client_addr = (
            websocket.remote_address
            if hasattr(websocket, "remote_address")
            else "unknown"
        )
        logger.info(f"New connection from {client_addr}, id={connection_id}")

        # Track last message time for debugging
        last_msg_time = [0]  # Use list to allow mutation in nested function

        session = self.session_manager.create_session()
        llm_client = LLMClient(self.config.llm, self.mcp_manager)
        client_tool_manager = self._create_client_tool_manager()

        conn = ClientConnection(
            websocket=websocket,
            session=session,
            config=self.config,
            llm_client=llm_client,
            mcp_manager=self.mcp_manager,
            client_tool_manager=client_tool_manager,
        )

        self._connections[connection_id] = conn

        try:
            await conn.send_json(
                create_status_message(
                    "connected", {"session_id": session.session_id}
                ).to_dict()
            )

            async for message in websocket:
                # Log raw message reception time for debugging
                import time
                recv_time = time.time()
                logger.info(f"[{connection_id}] WebSocket message received at {recv_time}")
                await self._handle_message(conn, message)

        except websockets.ConnectionClosed:
            logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            await conn.send_json(
                create_error_message(
                    ErrorCode.INTERNAL_ERROR, "Internal server error", str(e)
                ).to_dict()
            )
        finally:
            await llm_client.close()
            self.session_manager.end_session(session.session_id)
            if connection_id in self._connections:
                del self._connections[connection_id]
            logger.info(f"Connection cleanup: {connection_id}")

    async def _handle_message(self, conn: ClientConnection, message: Any) -> None:
        try:
            msg_data = parse_client_message(message)
            msg_type = msg_data.get("type")

            if msg_type == ClientMessageType.TEXT_INPUT.value:
                await self._handle_text_input(conn, msg_data)
            elif msg_type == ClientMessageType.CONFIGURE.value:
                await self._handle_configure(conn, msg_data)
            elif msg_type == ClientMessageType.START_SESSION.value:
                await self._handle_start_session(conn, msg_data)
            elif msg_type == ClientMessageType.END_SESSION.value:
                await self._handle_end_session(conn)
            elif msg_type == ClientMessageType.PING.value:
                await self._handle_ping(conn)
            elif msg_type == ClientMessageType.REGISTER_TOOLS.value:
                await self._handle_register_tools(conn, msg_data)
            elif msg_type == ClientMessageType.TOOL_RESULT.value:
                await self._handle_tool_result(conn, msg_data)
            else:
                await conn.send_json(
                    create_error_message(
                        ErrorCode.UNKNOWN_MESSAGE_TYPE,
                        f"Unknown message type: {msg_type}",
                    ).to_dict()
                )

        except json.JSONDecodeError:
            await conn.send_json(
                create_error_message(
                    ErrorCode.INVALID_MESSAGE, "Invalid JSON message"
                ).to_dict()
            )
        except Exception as e:
            import traceback

            logger.error(f"Error handling message: {e}\n{traceback.format_exc()}")
            await conn.send_json(
                create_error_message(
                    ErrorCode.INTERNAL_ERROR,
                    "Internal error processing message",
                    str(e),
                ).to_dict()
            )

    async def _handle_text_input(
        self, conn: ClientConnection, data: dict[str, Any]
    ) -> None:
        text = data.get("text", "")
        if not text:
            await conn.send_json(
                create_error_message(
                    ErrorCode.INVALID_MESSAGE, "text field is required"
                ).to_dict()
            )
            return

        session_id = data.get("session_id")
        if session_id:
            existing = self.session_manager.restore_session(session_id)
            if existing:
                conn.session = existing

        async with conn._lock:
            if conn.is_processing:
                await conn.send_json(
                    create_status_message(
                        "busy", {"message": "Already processing"}
                    ).to_dict()
                )
                return
            conn.is_processing = True

        try:
            await conn.send_json(create_status_message("processing").to_dict())

            conn.session.start_interaction()
            conn.session.add_message("user", text)

            # 根据配置决定是否使用上下文历史
            if conn.config.llm.enable_context:
                messages = conn.session.get_messages()
            else:
                # 只发送当前消息，不包含历史上下文
                messages = [{"role": "user", "content": text}]

            try:
                logger.info("Calling process_with_tools...")
                response = await conn.llm_client.process_with_tools(messages)
                logger.info("process_with_tools returned, extracting content and tool_calls...")
                logger.info(f"Response structure: choices={len(response.get('choices', []))}, "
                           f"has_message={bool(response.get('choices', [{}])[0].get('message')) if response.get('choices') else False}")
                content = conn.llm_client.extract_content(response)
                tool_calls = conn.llm_client.extract_tool_calls(response)
                logger.info(f"Extracted {len(tool_calls)} tool_calls from response")
                if not tool_calls:
                    logger.warning(f"WARNING: No tool_calls extracted! Response keys: {list(response.keys())}")
                    if response.get('choices'):
                        logger.warning(f"Choice keys: {list(response['choices'][0].keys())}")
                        if response['choices'][0].get('message'):
                            logger.warning(f"Message keys: {list(response['choices'][0]['message'].keys())}")
                            logger.warning(f"Message tool_calls: {response['choices'][0]['message'].get('tool_calls')}")

                # 检查是否有客户端工具调用
                client_tool_calls = []
                server_tool_calls = []

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    # 检查是否是客户端工具
                    is_client_tool = (
                        conn.client_tool_manager and
                        conn.client_tool_manager.has_tool(tool_name)
                    )

                    logger.info(f"Tool {tool_name}: is_client_tool={is_client_tool}, "
                               f"client_tool_manager exists: {conn.client_tool_manager is not None}")
                    if conn.client_tool_manager:
                        logger.info(f"  Client tools registered: {list(conn.client_tool_manager._tools.keys())}")

                    if is_client_tool:
                        client_tool_calls.append((tc, tool_name, args))
                    else:
                        server_tool_calls.append((tool_name, args))

                # 处理服务端工具调用通知
                for tool_name, args in server_tool_calls:
                    tool_msg = ToolCallMessage(
                        tool_name=tool_name,
                        arguments=args,
                        result=None,
                        success=True
                    )
                    await conn.send_json(tool_msg.to_dict())

                # 处理客户端工具调用
                # Store pending state for later continuation (don't block message loop)
                if client_tool_calls:
                    # Store the state needed to continue LLM after tools complete
                    choice = response.get("choices", [{}])[0]
                    assistant_msg = choice.get("message", {})

                    # Collect tool call IDs for tracking
                    pending_call_ids = []
                    tool_info_by_call_id = {}  # Map call_id -> (tool_call_id, tool_name, args)

                    for tc, tool_name, args in client_tool_calls:
                        tool_call_id = tc.get("id")
                        call_id, future = await conn.client_tool_manager.create_pending_call(
                            tool_name, args
                        )
                        pending_call_ids.append(call_id)
                        tool_info_by_call_id[call_id] = (tool_call_id, tool_name, args)

                        await conn.send_json(
                            ToolCallbackMessage(
                                call_id=call_id,
                                tool_name=tool_name,
                                arguments=args
                            ).to_dict()
                        )
                        logger.info(f"Sent tool callback to client: {tool_name} (call_id: {call_id})")

                    # Store pending state - will be resumed when tool_result messages arrive
                    conn._pending_llm_state = {
                        "messages": messages,
                        "assistant_msg": assistant_msg,
                        "pending_call_ids": pending_call_ids,
                        "tool_info_by_call_id": tool_info_by_call_id,
                        "original_text": text,
                    }
                    logger.info(f"Stored pending LLM state with {len(pending_call_ids)} pending calls. " +
                               "Message handler will return to allow receiving tool_result messages.")

                    # Send a "waiting_for_tools" status and return early
                    # Don't continue with LLM until tool results arrive
                    await conn.send_json(
                        create_status_message(
                            "waiting_for_tools",
                            {"pending_tools": len(pending_call_ids)}
                        ).to_dict()
                    )
                    # Return early - the LLM continuation will happen when tool_result arrives
                    return

                # 发送最终 LLM 响应 (for cases without client tools or client tools already handled)
                await conn.send_json(
                    LLMResponseMessage(
                        content=content,
                        tool_calls=[],
                        is_final=True
                    ).to_dict()
                )

                conn.session.add_message("assistant", content)
                conn.session.end_interaction(text, content)

            except Exception as e:
                import traceback
                logger.error(f"LLM processing error: {e}\n{traceback.format_exc()}")
                await conn.send_json(
                    create_error_message(
                        ErrorCode.LLM_ERROR,
                        "Failed to process request",
                        str(e)
                    ).to_dict()
                )

        finally:
            conn.is_processing = False

    async def _handle_configure(
        self, conn: ClientConnection, data: dict[str, Any]
    ) -> None:
        temperature = data.get("temperature")
        max_tokens = data.get("max_tokens")
        enable_context = data.get("enable_context")

        if temperature is not None:
            conn.config.llm.temperature = float(temperature)
        if max_tokens is not None:
            conn.config.llm.max_tokens = int(max_tokens)
        if enable_context is not None:
            conn.config.llm.enable_context = bool(enable_context)

        await conn.send_json(
            create_status_message(
                "configured",
                {
                    "temperature": conn.config.llm.temperature,
                    "max_tokens": conn.config.llm.max_tokens,
                    "enable_context": conn.config.llm.enable_context,
                },
            ).to_dict()
        )

    async def _handle_start_session(
        self, conn: ClientConnection, data: dict[str, Any]
    ) -> None:
        session_id = data.get("session_id")

        if session_id:
            existing = self.session_manager.restore_session(session_id)
            if existing:
                conn.session = existing
                await conn.send_json(
                    create_status_message(
                        "session_restored", {"session_id": session_id}
                    ).to_dict()
                )
                return

        await conn.send_json(
            create_status_message(
                "session_started", {"session_id": conn.session.session_id}
            ).to_dict()
        )

    async def _handle_end_session(self, conn: ClientConnection) -> None:
        self.session_manager.end_session(conn.session.session_id)
        new_session = self.session_manager.create_session()
        conn.session = new_session

        await conn.send_json(
            create_status_message(
                "session_ended", {"new_session_id": new_session.session_id}
            ).to_dict()
        )

    async def _handle_ping(self, conn: ClientConnection) -> None:
        await conn.send_json(PongMessage().to_dict())

    async def _handle_register_tools(
        self, conn: ClientConnection, data: dict[str, Any]
    ) -> None:
        """处理客户端工具注册请求"""
        if not conn.client_tool_manager:
            await conn.send_json(
                create_error_message(
                    ErrorCode.TOOL_REGISTRATION_FAILED,
                    "Client tools are not enabled"
                ).to_dict()
            )
            return

        tools = data.get("tools", [])
        if not isinstance(tools, list):
            await conn.send_json(
                create_error_message(
                    ErrorCode.INVALID_MESSAGE,
                    "tools field must be an array"
                ).to_dict()
            )
            return

        results = await conn.client_tool_manager.register_tools(tools)

        # 更新 LLM 客户端的工具列表
        conn.llm_client.set_client_tools(conn.client_tool_manager)

        await conn.send_json(
            ToolsRegisteredMessage(
                count=len([r for r in results if r["status"] == "registered"]),
                tools=results
            ).to_dict()
        )

    async def _handle_tool_result(
        self, conn: ClientConnection, data: dict[str, Any]
    ) -> None:
        """处理客户端工具执行结果"""
        import time
        logger.info(f"Received tool_result message at {time.time()}: {data}")

        if not conn.client_tool_manager:
            await conn.send_json(
                create_error_message(
                    ErrorCode.INVALID_MESSAGE,
                    "Client tools are not enabled"
                ).to_dict()
            )
            return

        call_id = data.get("call_id")
        if not call_id:
            await conn.send_json(
                create_error_message(
                    ErrorCode.INVALID_MESSAGE,
                    "call_id field is required"
                ).to_dict()
            )
            return

        success = data.get("success", True)
        result = data.get("result")
        error = data.get("error")

        logger.info(f"Handling tool result: call_id={call_id}, success={success}, result={result}")

        handled = await conn.client_tool_manager.handle_tool_result(
            call_id=call_id,
            result=result,
            success=success,
            error=error
        )

        logger.info(f"Tool result handled: {handled}")

        if not handled:
            logger.warning(f"Failed to handle tool result for call_id: {call_id}")
            await conn.send_json(
                create_error_message(
                    ErrorCode.INVALID_MESSAGE,
                    f"Invalid or expired call_id: {call_id}"
                ).to_dict()
            )
            return

        # Check if there's a pending LLM continuation waiting for this tool result
        if conn._pending_llm_state and call_id in conn._pending_llm_state.get("pending_call_ids", []):
            await self._continue_llm_after_tool_result(conn, call_id, result, success)

    async def _continue_llm_after_tool_result(
        self, conn: ClientConnection, call_id: str, result: Any, success: bool
    ) -> None:
        """Continue LLM processing after receiving tool result"""
        state = conn._pending_llm_state
        if not state:
            logger.warning(f"No pending LLM state for tool_result: {call_id}")
            return

        # Initialize completed_call_ids if not exists
        if "completed_call_ids" not in state:
            state["completed_call_ids"] = []

        # Mark this call as completed
        state["completed_call_ids"].append(call_id)
        state.setdefault("tool_results", {})[call_id] = result if success else {"error": result}

        # Get tool info for sending notification
        _tool_call_id, tool_name, args = state["tool_info_by_call_id"][call_id]

        # Send tool call completion notification
        if success:
            tool_msg = ToolCallMessage(
                tool_name=tool_name,
                arguments=args,
                result=result,
                success=True
            )
        else:
            tool_msg = ToolCallMessage(
                tool_name=tool_name,
                arguments=args,
                result={"error": result},
                success=False
            )
        await conn.send_json(tool_msg.to_dict())

        # Check if all pending calls are complete
        pending_ids = state["pending_call_ids"]
        completed_ids = state["completed_call_ids"]

        if len(completed_ids) == len(pending_ids):
            logger.info(f"All {len(pending_ids)} client tools completed. Continuing LLM call...")
            await self._finalize_llm_with_client_tools(conn, state)
        else:
            logger.info(f"Tool result received ({len(completed_ids)}/{len(pending_ids)} complete). Waiting for more...")

    async def _finalize_llm_with_client_tools(
        self, conn: ClientConnection, state: dict[str, Any]
    ) -> None:
        """Finalize LLM call after all client tools complete"""
        try:
            messages = state["messages"]
            assistant_msg = state["assistant_msg"]
            pending_call_ids = state["pending_call_ids"]
            tool_info_by_call_id = state["tool_info_by_call_id"]
            original_text = state["original_text"]

            # Build tool result messages
            tool_continuation = [assistant_msg]

            for call_id in pending_call_ids:
                tool_call_id, tool_name, _args = tool_info_by_call_id[call_id]
                result = state["tool_results"][call_id]

                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False)
                }
                tool_continuation.append(tool_result_msg)

            logger.info(f"Continuing LLM with {len(tool_continuation)} messages")

            # Get final response from LLM
            final_response = await conn.llm_client.continue_with_client_tool_results(
                messages, tool_continuation
            )
            content = conn.llm_client.extract_content(final_response)

            logger.info(f"Final LLM response content: {content}")

            # Send final LLM response
            await conn.send_json(
                LLMResponseMessage(
                    content=content,
                    tool_calls=[],
                    is_final=True
                ).to_dict()
            )

            conn.session.add_message("assistant", content)
            conn.session.end_interaction(original_text, content)

        except Exception as e:
            import traceback
            logger.error(f"Error finalizing LLM with client tools: {e}\n{traceback.format_exc()}")
            await conn.send_json(
                create_error_message(
                    ErrorCode.LLM_ERROR,
                    "Failed to process tool results",
                    str(e)
                ).to_dict()
            )
        finally:
            # Clear pending state
            conn._pending_llm_state = None


async def run_http_server(config: Config, port: int = 9401):
    app = web.Application()

    cloud_dir = Path(__file__).parent.parent

    async def serve_test_client(request: web.Request) -> web.Response:
        test_client_path = cloud_dir / "test_client.html"
        if test_client_path.exists():
            return web.Response(
                text=test_client_path.read_text(), content_type="text/html"
            )
        return web.Response(text="Test client not found", status=404)

    async def health_check(request: web.Request) -> web.Response:
        return web.Response(
            text=json.dumps({"status": "healthy", "service": "cloud-agent-v1.1"}),
            content_type="application/json",
        )

    app.router.add_get("/", serve_test_client)
    app.router.add_get("/test_client.html", serve_test_client)
    app.router.add_get("/health", health_check)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.server.host, port)
    await site.start()

    logger.info(f"HTTP server started on {config.server.host}:{port}")

    return runner


async def main():
    config = Config.from_env()
    server = CloudServer(config)

    http_port = int(os.getenv("HTTP_PORT", "9401"))
    http_runner = await run_http_server(config, http_port)

    try:
        await server.start()
    finally:
        await http_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
