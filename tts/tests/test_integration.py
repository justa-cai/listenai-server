"""Integration tests for VoxCPM TTS Server."""

import pytest
import asyncio
import json
import struct
from typing import Optional

from src.config import ServerConfig, ModelConfig, Config
from src.server import VoxCPMWebSocketServer


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def send(self, data):
        """Store sent message."""
        self.sent_messages.append(data)

    async def close(self):
        """Mark as closed."""
        self.closed = True


@pytest.fixture
def server_config():
    """Create test server configuration."""
    return ServerConfig(
        host="localhost",
        port=9300,
        max_concurrent_requests=2,
        max_connections=10,
        ping_interval=30,
        ping_timeout=10
    )


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        model_name="test-model",
        device="cpu"
    )


@pytest.fixture
def config(server_config, model_config):
    """Create full test configuration."""
    return Config(server=server_config, model=model_config)


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 9300
        assert config.max_concurrent_requests == 10
        assert config.max_connections == 100

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment."""
        monkeypatch.setenv("TTS_HOST", "127.0.0.1")
        monkeypatch.setenv("TTS_PORT", "9999")
        monkeypatch.setenv("TTS_MAX_CONCURRENT", "5")

        config = ServerConfig()
        config.host = "127.0.0.1"
        config.port = 9999
        config.max_concurrent_requests = 5

        assert config.host == "127.0.0.1"
        assert config.port == 9999
        assert config.max_concurrent_requests == 5


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.model_name == "openbmb/VoxCPM-0.5B"
        assert config.device == "cuda"
        assert config.default_cfg_value == 2.0


class TestConfig:
    """Tests for Config."""

    def test_config_creation(self, server_config, model_config):
        """Test creating full configuration."""
        config = Config(server=server_config, model=model_config)
        assert config.server is server_config
        assert config.model is model_config


class TestBinaryFrameParsing:
    """Tests for binary frame parsing."""

    def test_parse_valid_frame(self):
        """Test parsing a valid binary frame."""
        from examples.client_example import parse_binary_frame

        metadata = json.dumps({
            "request_id": "test-123",
            "sequence": 0,
            "sample_rate": 24000
        }).encode('utf-8')

        audio_data = b'\x00\x01' * 100  # Some dummy audio

        # Build frame
        frame = bytearray()
        frame.extend([0xAA, 0x55])  # Magic
        frame.append(0x01)  # Message type
        frame.append(0x00)  # Reserved
        frame.extend(len(metadata).to_bytes(4, 'big'))  # Metadata length
        frame.extend(metadata)  # Metadata
        frame.extend(len(audio_data).to_bytes(4, 'big'))  # Payload length
        frame.extend(audio_data)  # Audio data

        # Parse
        parsed_metadata, parsed_audio = parse_binary_frame(bytes(frame))

        assert parsed_metadata["request_id"] == "test-123"
        assert parsed_metadata["sequence"] == 0
        assert parsed_metadata["sample_rate"] == 24000
        assert parsed_audio == audio_data

    def test_parse_invalid_magic(self):
        """Test parsing frame with invalid magic number."""
        from examples.client_example import parse_binary_frame

        frame = b'\x00\x00\x01\x00\x00\x00\x00\x00'  # Invalid magic

        with pytest.raises(ValueError, match="Invalid magic number"):
            parse_binary_frame(frame)


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_pcm16_conversion(self):
        """Test PCM16 conversion roundtrip."""
        from src.audio_processor import AudioProcessor
        import numpy as np

        processor = AudioProcessor()
        original = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)

        # Convert to PCM16
        pcm_bytes = processor.to_pcm16(original)

        # Convert back
        recovered = processor.from_pcm16(pcm_bytes)

        # Check values (allowing for some quantization error)
        assert len(recovered) == len(original)
        assert abs(recovered[0]) < 0.01  # Near zero
        assert abs(recovered[2] - 1.0) < 0.01  # Near max
        assert abs(recovered[4] - (-1.0)) < 0.01  # Near min


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_and_get_session(self):
        """Test creating and retrieving a session."""
        from src.session import SessionManager

        manager = SessionManager()
        session = manager.create_session(
            request_id="test-123",
            connection_id="conn-456",
            params={"text": "Hello"}
        )

        assert session.request_id == "test-123"
        assert session.connection_id == "conn-456"
        assert session.state == "created"

        # Retrieve
        retrieved = manager.get_session("test-123")
        assert retrieved is session

    def test_remove_session(self):
        """Test removing a session."""
        from src.session import SessionManager

        manager = SessionManager()
        manager.create_session(
            request_id="test-123",
            connection_id="conn-456",
            params={}
        )

        removed = manager.remove_session("test-123")
        assert removed is not None
        assert removed.request_id == "test-123"

        # Should be gone
        assert manager.get_session("test-123") is None

    def test_cleanup_connection_sessions(self):
        """Test cleaning up all sessions for a connection."""
        from src.session import SessionManager

        manager = SessionManager()
        manager.create_session("req-1", "conn-1", {})
        manager.create_session("req-2", "conn-1", {})
        manager.create_session("req-3", "conn-2", {})

        removed = manager.cleanup_connection_sessions("conn-1")

        assert len(removed) == 2
        assert manager.get_session("req-1") is None
        assert manager.get_session("req-2") is None
        assert manager.get_session("req-3") is not None


class TestTaskQueue:
    """Tests for TaskQueue."""

    @pytest.mark.asyncio
    async def test_queue_status(self):
        """Test queue status reporting."""
        from src.queue import TaskQueue

        queue = TaskQueue(max_concurrent=5)
        status = queue.get_status()

        assert status["pending"] == 0
        assert status["running"] == 0
        assert status["max_concurrent"] == 5
        assert status["available_slots"] == 5
