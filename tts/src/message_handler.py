"""Message handler for WebSocket communications."""

import logging
import time
import asyncio
from typing import Optional

from .connection import Connection
from .session import SessionManager
from .tasks import TTSRequestTask
from .validators import TTSRequestValidator
from .errors import (
    ValidationError,
    InvalidJSONError,
    UnknownMessageTypeError,
)

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handles incoming WebSocket messages."""

    def __init__(self, model_config, queue, voice_manager=None, server_config=None):
        """
        Initialize the message handler.

        Args:
            model_config: Model configuration
            queue: Task queue for TTS requests
            voice_manager: Optional voice manager for voice cloning
            server_config: Optional server configuration
        """
        self.model_config = model_config
        self.queue = queue
        self.validator = TTSRequestValidator(model_config)
        self.voice_manager = voice_manager
        self.server_config = server_config

    def set_voice_manager(self, voice_manager):
        """Set the voice manager."""
        self.voice_manager = voice_manager

    async def handle(
        self,
        connection: Connection,
        message: dict,
        session_manager: SessionManager
    ) -> Optional[str]:
        """
        Handle an incoming message.

        Args:
            connection: Connection object
            message: Parsed message dictionary
            session_manager: Session manager

        Returns:
            Optional response message (JSON string)
        """
        msg_type = message.get("type")

        if msg_type == "tts_request":
            await self._handle_tts_request(connection, message, session_manager)
        elif msg_type == "cancel":
            return await self._handle_cancel(connection, message, session_manager)
        elif msg_type == "ping":
            return await self._handle_ping(connection, message)
        else:
            raise UnknownMessageTypeError(f"Unknown message type: {msg_type}")

        return None

    async def _handle_tts_request(
        self,
        connection: Connection,
        message: dict,
        session_manager: SessionManager
    ) -> None:
        """Handle a TTS request."""
        request_id = message.get("request_id")
        params = message.get("params", {})

        # Record current request ID for error handling
        connection.metadata["current_request_id"] = request_id

        # Resolve voice_id to prompt_wav_path if provided
        if "voice_id" in params:
            voice_id = params["voice_id"]
            if self.voice_manager:
                voice = self.voice_manager.get_voice(voice_id)
                if voice:
                    params["prompt_wav_path"] = voice.audio_path
                    params["prompt_text"] = voice.sample_text
                    logger.info(f"Using voice {voice_id} ({voice.name}) for TTS request")
                else:
                    await connection.send_error(
                        "VOICE_NOT_FOUND",
                        f"Voice '{voice_id}' not found",
                        {"voice_id": voice_id}
                    )
                    return
            else:
                await connection.send_error(
                    "VOICE_MANAGER_NOT_AVAILABLE",
                    "Voice manager not initialized"
                )
                return

        # Validate request
        validation_result = self.validator.validate(params)
        if not validation_result.is_valid:
            await connection.send_error(
                "INVALID_PARAMS",
                validation_result.error_message,
                {"errors": validation_result.errors}
            )
            return

        # Create session
        session = session_manager.create_session(
            request_id=request_id,
            connection_id=connection.id,
            params=params
        )

        # Create and queue task
        task = TTSRequestTask(
            session=session,
            model_config=self.model_config,
            connection=connection,
            server_config=self.server_config
        )

        # Add to queue
        try:
            await self.queue.put(task)
            session.state = "queued"
            await connection.send_progress(
                request_id,
                "queued",
                0.0,
                "Request queued"
            )
        except asyncio.QueueFull:
            await connection.send_error(
                "QUEUE_FULL",
                "Request queue is full, please try again later"
            )
            session_manager.remove_session(request_id)

    async def _handle_cancel(
        self,
        connection: Connection,
        message: dict,
        session_manager: SessionManager
    ) -> Optional[str]:
        """Handle a cancel request."""
        request_id = message.get("request_id")

        session = session_manager.get_session(request_id)
        if session:
            session.cancelled = True
            session.state = "cancelled"

        # Send complete message with cancelled flag
        await connection.send_complete(request_id, {"cancelled": True})

        # Clean up session
        session_manager.remove_session(request_id)

        return None

    async def _handle_ping(
        self,
        connection: Connection,
        message: dict
    ) -> str:
        """Handle a ping message."""
        timestamp = message.get("timestamp", int(time.time()))
        connection.update_ping_time()
        return connection.send_pong(timestamp)
