import asyncio
import json
import struct
import uuid
from typing import Callable, Optional, Any
import logging
import websockets

from .config import TTSConfig

logger = logging.getLogger(__name__)


class TTSClient:
    def __init__(self, config: TTSConfig):
        self.config = config
        self._ws: Any = None
        self._connected = False
        self._connecting = False
        self._audio_handlers: list[Callable[[bytes, bool], Any]] = []
        self._receive_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._current_request_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._ws is None:
            return False
        try:
            from websockets.protocol import State

            return self._ws.state == State.OPEN
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"TTS connection check failed: {e}")
            return self._connected

    async def connect(self) -> bool:
        async with self._lock:
            if self.is_connected:
                return True

            if self._connecting:
                logger.warning("TTS connection already in progress")
                return False

            self._connecting = True

            try:
                logger.info(f"Connecting to TTS service: {self.config.service_url}")
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
                logger.info("Connected to TTS service")
                return True

            except asyncio.TimeoutError:
                logger.error("TTS connection timeout")
                self._connecting = False
                return False
            except Exception as e:
                logger.error(f"Failed to connect to TTS service: {e}")
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
                    logger.error(f"Error closing TTS connection: {e}")
                self._ws = None

            self._connected = False
            logger.info("Disconnected from TTS service")

    async def reconnect(self) -> bool:
        logger.info("Reconnecting to TTS service...")
        await self.disconnect()
        await asyncio.sleep(0.5)
        return await self.connect()

    def on_audio(self, handler: Callable[[bytes, bool], Any]) -> None:
        self._audio_handlers.append(handler)

    def remove_handler(self, handler: Callable[[bytes, bool], Any]) -> None:
        if handler in self._audio_handlers:
            self._audio_handlers.remove(handler)

    async def _receive_loop(self) -> None:
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                if isinstance(message, bytes):
                    await self._handle_binary_frame(message)
                elif isinstance(message, str):
                    await self._handle_json_message(message)
            except websockets.ConnectionClosed:
                logger.warning("TTS connection closed")
                self._connected = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TTS receive loop: {e}")

    async def _handle_binary_frame(self, data: bytes) -> None:
        try:
            if len(data) < 8:
                logger.warning(f"TTS frame too short: {len(data)} bytes")
                return

            magic = struct.unpack(">H", data[0:2])[0]
            if magic != 0xAA55:
                logger.warning(f"Invalid TTS frame magic: {hex(magic)}")
                return

            msg_type = data[2]
            metadata_length = struct.unpack(">I", data[4:8])[0]

            if len(data) < 8 + metadata_length + 4:
                logger.warning("TTS frame incomplete")
                return

            metadata_json = data[8 : 8 + metadata_length].decode("utf-8")
            metadata = json.loads(metadata_json)

            payload_offset = 8 + metadata_length
            payload_length = struct.unpack(
                ">I", data[payload_offset : payload_offset + 4]
            )[0]

            audio_start = payload_offset + 4
            audio_end = audio_start + payload_length
            audio_data = data[audio_start:audio_end]

            is_final = metadata.get("is_final", False)

            if audio_data:
                logger.info(
                    f"TTS audio chunk: {len(audio_data)} bytes, is_final={is_final}"
                )
                await self._handle_audio(audio_data, is_final)

        except Exception as e:
            logger.error(f"Failed to parse TTS frame: {e}")

    async def _handle_json_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            logger.debug(f"TTS JSON message: {msg_type} - {data}")

            if msg_type == "complete":
                logger.info(f"TTS synthesis complete: {data.get('request_id')}")
                await self._handle_audio(b"", True)
            elif msg_type == "error":
                error = data.get("error", {})
                logger.error(
                    f"TTS error: code={error.get('code')}, message={error.get('message')}, raw={data}"
                )
            elif msg_type == "progress":
                logger.info(
                    f"TTS progress: {data.get('state')} - {data.get('message')}"
                )

        except json.JSONDecodeError:
            logger.warning(f"Invalid TTS JSON message: {message[:200]}")

    async def _handle_audio(self, audio_data: bytes, is_final: bool) -> None:
        for handler in self._audio_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(audio_data, is_final)
                else:
                    handler(audio_data, is_final)
            except Exception as e:
                logger.error(f"Error in TTS audio handler: {e}")

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bool:
        if not self.is_connected:
            logger.warning("Cannot synthesize: TTS not connected")
            return False

        self._current_request_id = str(uuid.uuid4())

        params: dict[str, Any] = {
            "text": text,
            "mode": self.config.mode,
            "cfg_value": self.config.cfg_value,
            "inference_timesteps": self.config.inference_timesteps,
        }

        effective_voice_id = voice_id or self.config.voice_id
        if effective_voice_id:
            params["voice_id"] = effective_voice_id

        payload = {
            "type": "tts_request",
            "request_id": self._current_request_id,
            "params": params,
        }

        try:
            await self._ws.send(json.dumps(payload))
            logger.info(
                f"Sent TTS request: {self._current_request_id}, text: {text[:50]}..."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send TTS request: {e}")
            self._connected = False
            return False

        self._current_request_id = str(uuid.uuid4())

        payload = {
            "type": "tts_request",
            "request_id": self._current_request_id,
            "params": {
                "text": text,
                "mode": self.config.mode,
                "voice_id": voice_id or self.config.voice_id,
                "cfg_value": self.config.cfg_value,
                "inference_timesteps": self.config.inference_timesteps,
            },
        }

        try:
            await self._ws.send(json.dumps(payload))
            logger.info(
                f"Sent TTS request: {self._current_request_id}, text: {text[:50]}..."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send TTS request: {e}")
            self._connected = False
            return False

    async def send_command(self, command: dict[str, Any]) -> bool:
        if not self.is_connected:
            return False

        try:
            await self._ws.send(json.dumps(command))
            return True
        except Exception as e:
            logger.error(f"Failed to send TTS command: {e}")
            return False
