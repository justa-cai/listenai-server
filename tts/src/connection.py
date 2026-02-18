"""WebSocket connection management."""

import uuid
import json
import time
import websockets.server
from datetime import datetime
from typing import Optional

from .errors import VoxCPMError


class Connection:
    """Wrapper for a WebSocket connection."""

    # Frame type constants
    FRAME_TYPE_STREAMING_CHUNK = 0x01
    FRAME_TYPE_NON_STREAMING = 0x02
    FRAME_TYPE_METADATA_ONLY = 0x03

    def __init__(self, websocket: websockets.server.WebSocketServerProtocol):
        """
        Initialize connection.

        Args:
            websocket: WebSocket connection object
        """
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.created_at = datetime.now()
        self.last_ping = datetime.now()
        self.metadata = {}
        self.remote_address = websocket.remote_address

    async def send(self, data: str | bytes):
        """
        Send data through the WebSocket.

        Args:
            data: String or bytes to send
        """
        await self.websocket.send(data)

    async def send_json(self, obj: dict):
        """
        Send a JSON message.

        Args:
            obj: Dictionary to serialize as JSON
        """
        await self.send(json.dumps(obj))

    async def send_binary_frame(
        self,
        msg_type: int,
        metadata: dict,
        audio_data: bytes
    ):
        """
        Send a binary frame with metadata and audio data.

        Frame structure:
        - Magic (2 bytes): 0xAA 0x55
        - Message Type (1 byte)
        - Reserved (1 byte): 0x00
        - Metadata Length (4 bytes, big endian)
        - Metadata JSON (variable length)
        - Payload Length (4 bytes, big endian)
        - Audio Payload (variable length, PCM 16-bit)

        Args:
            msg_type: Message type (0x01 for streaming, 0x02 for non-streaming)
            metadata: Metadata dictionary
            audio_data: Raw audio bytes (PCM 16-bit, little-endian)
        """
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata_json)

        # Check audio data length is even (PCM 16-bit)
        if len(audio_data) % 2 != 0:
            raise ValueError(f"Audio data length must be even for PCM 16-bit, got {len(audio_data)}")

        frame = bytearray()
        # Magic number
        frame.extend([0xAA, 0x55])
        # Message type
        frame.append(msg_type)
        # Reserved
        frame.append(0x00)
        # Metadata length (big endian)
        frame.extend(metadata_length.to_bytes(4, 'big'))
        # Metadata JSON
        frame.extend(metadata_json)
        # Payload length (big endian)
        frame.extend(len(audio_data).to_bytes(4, 'big'))
        # Audio payload
        frame.extend(audio_data)

        await self.send(bytes(frame))

    async def send_progress(
        self,
        request_id: str,
        state: str,
        progress: float,
        message: str
    ):
        """
        Send a progress update message.

        Args:
            request_id: Request identifier
            state: Current state (queued, processing, generating, etc.)
            progress: Progress value (0.0 to 1.0)
            message: Progress message
        """
        await self.send_json({
            "type": "progress",
            "request_id": request_id,
            "state": state,
            "progress": progress,
            "message": message
        })

    async def send_complete(
        self,
        request_id: str,
        result: dict
    ):
        """
        Send a completion message.

        Args:
            request_id: Request identifier
            result: Result dictionary
        """
        await self.send_json({
            "type": "complete",
            "request_id": request_id,
            "result": result
        })

    async def send_error(
        self,
        code: str,
        message: str,
        details: dict = None
    ):
        """
        Send an error message.

        Args:
            code: Error code
            message: Error message
            details: Optional error details
        """
        error_obj = {
            "type": "error",
            "request_id": self.metadata.get("current_request_id"),
            "error": {
                "code": code,
                "message": message,
                "details": details or {}
            }
        }
        await self.send_json(error_obj)

    async def send_pong(self, timestamp: int):
        """
        Send a pong response.

        Args:
            timestamp: Original ping timestamp
        """
        await self.send_json({
            "type": "pong",
            "timestamp": timestamp,
            "server_time": int(time.time())
        })

    def update_ping_time(self):
        """Update the last ping time."""
        self.last_ping = datetime.now()

    def get_idle_time(self) -> float:
        """
        Get the idle time in seconds.

        Returns:
            Idle time in seconds
        """
        return (datetime.now() - self.last_ping).total_seconds()

    async def close(self):
        """Close the connection."""
        await self.websocket.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"Connection(id={self.id}, remote={self.remote_address})"
