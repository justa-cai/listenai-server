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
    ErrorCode,
    parse_client_message,
    ASRResultMessage,
    LLMResponseMessage,
    PongMessage,
    ToolCallMessage,
    create_error_message,
    create_status_message,
)
from .session import SessionManager, Session
from .audio_buffer import FrameBuffer
from .asr_client import ASRClient
from .llm_client import LLMClient
from .tts_client import TTSClient
from .mcp_manager import MCPToolManager

logger = logging.getLogger(__name__)


class ClientConnection:
    def __init__(
        self,
        websocket: Any,
        session: Session,
        config: Config,
        asr_client: ASRClient,
        llm_client: LLMClient,
        tts_client: TTSClient,
        mcp_manager: Optional[MCPToolManager],
    ):
        self.websocket = websocket
        self.session = session
        self.config = config
        self.asr_client = asr_client
        self.llm_client = llm_client
        self.tts_client = tts_client
        self.mcp_manager = mcp_manager
        self.frame_buffer = FrameBuffer(frame_size=config.audio.frame_size)
        self.is_processing = False
        self.audio_queue: list[bytes] = []
        self._lock = asyncio.Lock()
        self.last_asr_text = ""

    async def send_json(self, message: dict[str, Any]) -> None:
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def send_binary(self, data: bytes) -> None:
        try:
            await self.websocket.send(data)
        except Exception as e:
            logger.error(f"Failed to send binary data: {e}")


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
            )
            if self.config.mcp.enabled
            else None
        )

        self._connections: dict[str, ClientConnection] = {}
        self._connection_count = 0
        self._running = False

    async def start(self) -> None:
        self._running = True

        async with websockets.serve(
            self._handle_connection,
            self.config.server.host,
            self.config.server.port,
            ping_interval=self.config.server.ping_interval,
            ping_timeout=self.config.server.ping_timeout,
            max_size=10 * 1024 * 1024,
        ):
            logger.info(
                f"Cloud server started on {self.config.server.host}:{self.config.server.port}"
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

        session = self.session_manager.create_session()

        asr_client = ASRClient(self.config.asr)
        llm_client = LLMClient(self.config.llm, self.mcp_manager)
        tts_client = TTSClient(self.config.tts)

        conn = ClientConnection(
            websocket=websocket,
            session=session,
            config=self.config,
            asr_client=asr_client,
            llm_client=llm_client,
            tts_client=tts_client,
            mcp_manager=self.mcp_manager,
        )

        self._connections[connection_id] = conn

        asr_client.on_result(
            lambda result: asyncio.create_task(self._handle_asr_result(conn, result))
        )

        tts_client.on_audio(
            lambda data, is_final: asyncio.create_task(
                self._handle_tts_audio(conn, data, is_final)
            )
        )

        try:
            await asr_client.connect()
            await tts_client.connect()

            await conn.send_json(
                create_status_message(
                    "connected", {"session_id": session.session_id}
                ).to_dict()
            )

            async for message in websocket:
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
            await asr_client.disconnect()
            await tts_client.disconnect()
            await llm_client.close()
            self.session_manager.end_session(session.session_id)
            if connection_id in self._connections:
                del self._connections[connection_id]
            logger.info(f"Connection cleanup: {connection_id}")

    async def _handle_message(self, conn: ClientConnection, message: Any) -> None:
        try:
            if isinstance(message, bytes):
                await self._handle_audio_data(conn, message)
            else:
                msg_data = parse_client_message(message)
                msg_type = msg_data.get("type")

                if msg_type == ClientMessageType.AUDIO_DATA.value:
                    if "data" in msg_data and isinstance(msg_data["data"], bytes):
                        await self._handle_audio_data(conn, msg_data["data"])
                elif msg_type == ClientMessageType.CONFIGURE.value:
                    await self._handle_configure(conn, msg_data)
                elif msg_type == ClientMessageType.START_SESSION.value:
                    await self._handle_start_session(conn, msg_data)
                elif msg_type == ClientMessageType.END_SESSION.value:
                    await self._handle_end_session(conn)
                elif msg_type == ClientMessageType.PING.value:
                    await self._handle_ping(conn)
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

    async def _handle_audio_data(self, conn: ClientConnection, data: bytes) -> None:
        conn.frame_buffer.append(data)
        frames = conn.frame_buffer.extract_frames()

        if conn.is_processing:
            conn.audio_queue.extend(frames)
            return

        for frame in frames:
            if conn.asr_client.is_connected:
                await conn.asr_client.send_audio(frame)

    async def _handle_asr_result(
        self, conn: ClientConnection, result: dict[str, Any]
    ) -> None:
        text = result.get("text", "")
        is_final = result.get("is_final", False)
        is_speaking = result.get("is_speaking", False)
        segment_id = result.get("segment_id")

        await conn.send_json(
            ASRResultMessage(
                text=text,
                is_final=is_final,
                is_speaking=is_speaking,
                segment_id=segment_id,
            ).to_dict()
        )

        if is_final and text:
            conn.last_asr_text = text
            await self._process_user_input(conn, text)

    async def _process_user_input(self, conn: ClientConnection, text: str) -> None:
        async with conn._lock:
            if conn.is_processing:
                logger.debug("Already processing, queuing audio")
                return

            conn.is_processing = True

        try:
            await conn.send_json(create_status_message("processing").to_dict())

            conn.session.start_interaction()
            conn.session.add_message("user", text)

            messages = conn.session.get_messages()

            try:
                response = await conn.llm_client.process_with_tools(messages)
                content = conn.llm_client.extract_content(response)
                tool_calls = conn.llm_client.extract_tool_calls(response)

                tool_call_msgs = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    tool_msg = ToolCallMessage(
                        tool_name=tool_name, arguments=args, result=None, success=True
                    )
                    tool_call_msgs.append(tool_msg)
                    await conn.send_json(tool_msg.to_dict())

                await conn.send_json(
                    LLMResponseMessage(
                        content=content, tool_calls=[], is_final=True
                    ).to_dict()
                )

                conn.session.add_message("assistant", content)
                conn.session.end_interaction(text, content)

                if content and conn.tts_client.is_connected:
                    await conn.tts_client.synthesize(content)

            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                await conn.send_json(
                    create_error_message(
                        ErrorCode.LLM_ERROR, "Failed to process request", str(e)
                    ).to_dict()
                )

        finally:
            conn.is_processing = False

            if conn.audio_queue:
                frames = conn.audio_queue.copy()
                conn.audio_queue.clear()
                for frame in frames:
                    if conn.asr_client.is_connected:
                        await conn.asr_client.send_audio(frame)

    async def _handle_tts_audio(
        self, conn: ClientConnection, data: bytes, is_final: bool
    ) -> None:
        if data:
            logger.debug(f"Forwarding TTS audio to client: {len(data)} bytes")
            await conn.send_binary(data)

        if is_final:
            logger.info("TTS audio stream complete")
            await conn.send_json(create_status_message("audio_complete").to_dict())

    async def _handle_configure(
        self, conn: ClientConnection, data: dict[str, Any]
    ) -> None:
        language = data.get("language")
        voice_id = data.get("voice_id")

        if language:
            conn.config.server.log_level = language
        if voice_id:
            conn.config.tts.voice_id = voice_id

        await conn.send_json(
            create_status_message(
                "configured", {"language": language, "voice_id": voice_id}
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
        conn.last_asr_text = ""

        await conn.send_json(
            create_status_message(
                "session_ended", {"new_session_id": new_session.session_id}
            ).to_dict()
        )

    async def _handle_ping(self, conn: ClientConnection) -> None:
        await conn.send_json(PongMessage().to_dict())


async def run_http_server(config: Config, port: int = 8080):
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
            text=json.dumps({"status": "healthy", "service": "cloud-agent"}),
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
