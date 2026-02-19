import asyncio
import json
from typing import Callable, Optional, Any
import logging
import websockets

from .config import ASRConfig

logger = logging.getLogger(__name__)


class ASRClient:
    def __init__(self, config: ASRConfig):
        self.config = config
        self._ws: Any = None
        self._connected = False
        self._connecting = False
        self._result_handlers: list[Callable[[dict[str, Any]], Any]] = []
        self._receive_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._ws is None:
            return False
        try:
            from websockets.protocol import State

            return self._ws.state == State.OPEN
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"ASR connection check failed: {e}")
            return self._connected
        try:
            return self._ws.open
        except (AttributeError, RuntimeError) as e:
            logger.warning(
                f"ASR connection check failed: {e}, _ws type: {type(self._ws)}"
            )
            return False

    async def connect(self) -> bool:
        async with self._lock:
            if self.is_connected:
                return True

            if self._connecting:
                logger.warning("ASR connection already in progress")
                return False

            self._connecting = True

            try:
                logger.info(f"Connecting to ASR service: {self.config.service_url}")
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.config.service_url,
                        ping_interval=30,
                        ping_timeout=10,
                    ),
                    timeout=self.config.timeout,
                )
                self._connected = True
                self._connecting = False

                self._receive_task = asyncio.create_task(self._receive_loop())
                logger.info("Connected to ASR service")
                return True

            except asyncio.TimeoutError:
                logger.error("ASR connection timeout")
                self._connecting = False
                return False
            except Exception as e:
                logger.error(f"Failed to connect to ASR service: {e}")
                self._connecting = False
                return False

    async def disconnect(self) -> None:
        async with self._lock:
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                self._receive_task = None

            if self._ws:
                try:
                    await self._ws.close()
                except Exception as e:
                    logger.error(f"Error closing ASR connection: {e}")
                self._ws = None

            self._connected = False
            logger.info("Disconnected from ASR service")

    async def reconnect(self) -> bool:
        logger.info("Reconnecting to ASR service...")
        await self.disconnect()
        await asyncio.sleep(0.5)
        return await self.connect()

    def on_result(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._result_handlers.append(handler)

    def remove_handler(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        if handler in self._result_handlers:
            self._result_handlers.remove(handler)

    async def _receive_loop(self) -> None:
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                if isinstance(message, str):
                    try:
                        result = json.loads(message)
                        await self._handle_result(result)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode ASR result: {e}")
            except websockets.ConnectionClosed:
                logger.warning("ASR connection closed")
                self._connected = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ASR receive loop: {e}")

    async def _handle_result(self, result: dict[str, Any]) -> None:
        for handler in self._result_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(result)
                else:
                    handler(result)
            except Exception as e:
                logger.error(f"Error in ASR result handler: {e}")

    async def send_audio(self, audio_data: bytes) -> bool:
        if not self.is_connected:
            logger.warning("Cannot send audio: ASR not connected")
            return False

        try:
            await self._ws.send(audio_data)
            return True
        except Exception as e:
            logger.error(f"Failed to send audio to ASR: {e}")
            self._connected = False
            return False

    async def send_command(self, command: dict[str, Any]) -> bool:
        if not self.is_connected:
            logger.warning("Cannot send command: ASR not connected")
            return False

        try:
            await self._ws.send(json.dumps(command))
            return True
        except Exception as e:
            logger.error(f"Failed to send command to ASR: {e}")
            return False

    async def reset(self) -> bool:
        return await self.send_command({"type": "reset"})

    async def ping(self) -> bool:
        return await self.send_command({"type": "ping"})
