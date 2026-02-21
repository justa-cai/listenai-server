"""WebSocket server for VoxCPM TTS."""

import asyncio
import logging
from typing import Set
from dataclasses import dataclass, field

import websockets.server
from websockets.exceptions import ConnectionClosed

from .connection import Connection
from .session import SessionManager
from .message_handler import MessageHandler
from .queue import TaskQueue
from .config import ServerConfig, ModelConfig
from .metrics import metrics_manager

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    """Server state."""
    connections: Set[Connection] = field(default_factory=set)
    active_requests: dict = field(default_factory=dict)


class VoxCPMWebSocketServer:
    """
    VoxCPM WebSocket TTS Server.

    Handles WebSocket connections and routes TTS requests.
    """

    def __init__(self, server_config: ServerConfig, model_config: ModelConfig):
        """
        Initialize the server.

        Args:
            server_config: Server configuration
            model_config: Model configuration
        """
        self.server_config = server_config
        self.model_config = model_config
        self.state = ServerState()
        self.session_manager = SessionManager()
        self.queue = TaskQueue(max_concurrent=server_config.max_concurrent_requests)
        self.message_handler = MessageHandler(model_config, self.queue, server_config=server_config)
        self._running = False
        self._server = None
        self._metrics_task = None

    async def serve(self, host: str = None, port: int = None) -> None:
        """
        Start the WebSocket server.

        Args:
            host: Host to bind to (defaults to config)
            port: Port to bind to (defaults to config)
        """
        host = host or self.server_config.host
        port = port or self.server_config.port

        self._running = True

        # Start task queue workers
        await self.queue.start(num_workers=self.server_config.max_concurrent_requests)

        # Start metrics reporting if enabled
        if self.server_config.metrics_enabled:
            self._metrics_task = asyncio.create_task(self._report_metrics())

        logger.info(f"Starting VoxCPM TTS server on {host}:{port}")

        async with websockets.server.serve(
            self._handle_connection,
            host,
            port,
            ping_interval=self.server_config.ping_interval,
            ping_timeout=self.server_config.ping_timeout,
            max_size=self.server_config.max_message_size,
            compression=None,
            process_request=self._process_request
        ) as server:
            self._server = server
            logger.info(f"Server listening on ws://{host}:{port}/tts")
            await self._run_forever()

    async def _run_forever(self) -> None:
        """Keep server running."""
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the server."""
        logger.info("Stopping server...")
        self._running = False

        try:
            # Stop task queue with timeout
            await asyncio.wait_for(self.queue.stop(), timeout=0.8)
        except asyncio.TimeoutError:
            logger.warning("Queue stop timed out")

        # Stop metrics task
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await asyncio.wait_for(self._metrics_task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Close all connections with timeout
        for connection in list(self.state.connections):
            try:
                await asyncio.wait_for(connection.close(), timeout=0.5)
            except asyncio.TimeoutError:
                logger.warning(f"Connection {connection.id} close timed out")

        logger.info("Server stopped")

    async def _process_request(self, path: str, request_headers: dict) -> None:
        """Process incoming HTTP request before WebSocket upgrade."""
        # Validate path
        if path not in ("/tts", "/"):
            logger.warning(f"Invalid path: {path}")
            raise ValueError(f"Invalid path: {path}")

    async def _handle_connection(
        self,
        websocket: websockets.server.WebSocketServerProtocol,
        path: str
    ) -> None:
        """Handle a new WebSocket connection."""
        connection = Connection(websocket)
        self.state.connections.add(connection)

        logger.info(f"New connection: {connection.id} from {connection.remote_address}")

        # Update metrics
        if self.server_config.metrics_enabled:
            metrics_manager.active_connections.inc()

        try:
            await self._handle_messages(connection)
        except ConnectionClosed:
            logger.info(f"Connection closed: {connection.id}")
        except Exception as e:
            logger.exception(f"Error handling connection {connection.id}: {e}")
        finally:
            await self._cleanup_connection(connection)

    async def _handle_messages(self, connection: Connection) -> None:
        """Handle messages from a connection."""
        async for raw_message in connection.websocket:
            try:
                if isinstance(raw_message, bytes):
                    # Binary messages from client are not supported
                    await connection.send_error(
                        "UNSUPPORTED_FORMAT",
                        "Binary messages from client are not supported"
                    )
                    continue

                # Parse JSON message
                import json
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    await connection.send_error("INVALID_JSON", "Invalid JSON format")
                    continue

                # Handle message
                response = await self.message_handler.handle(
                    connection,
                    message,
                    self.session_manager
                )

                if response:
                    await connection.send(response)

            except Exception as e:
                logger.exception(f"Error handling message: {e}")
                await connection.send_error("INTERNAL_ERROR", str(e))

    async def _cleanup_connection(self, connection: Connection) -> None:
        """Clean up a closed connection."""
        self.state.connections.discard(connection)

        # Update metrics
        if self.server_config.metrics_enabled:
            metrics_manager.active_connections.dec()

        # Clean up sessions
        removed_sessions = self.session_manager.cleanup_connection_sessions(connection.id)
        logger.info(f"Cleaned up {len(removed_sessions)} sessions for connection {connection.id}")

    async def _report_metrics(self) -> None:
        """Periodically report metrics."""
        while self._running:
            try:
                if self.server_config.metrics_enabled:
                    metrics_manager.queue_length.set(self.queue.pending_count)
                    metrics_manager.running_tasks.set(self.queue.running_count)

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")

    def get_status(self) -> dict:
        """Get server status."""
        return {
            "running": self._running,
            "connections": len(self.state.connections),
            "sessions": self.session_manager.session_count,
            "queue": self.queue.get_status(),
        }
