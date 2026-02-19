from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16

    @property
    def bytes_per_sample(self) -> int:
        return self.bits_per_sample // 8

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.bytes_per_sample


class AudioBuffer:
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        max_duration_seconds: float = 30.0,
    ):
        self.config = config or AudioConfig()
        self.max_duration_seconds = max_duration_seconds
        self._buffer: bytearray = bytearray()

    def append(self, data: bytes) -> int:
        max_bytes = int(self.max_duration_seconds * self.config.bytes_per_second)
        self._buffer.extend(data)

        if len(self._buffer) > max_bytes:
            excess = len(self._buffer) - max_bytes
            self._buffer = self._buffer[excess:]
            logger.warning(f"Audio buffer overflow, dropped {excess} bytes")

        return len(data)

    def get_bytes(self) -> bytes:
        return bytes(self._buffer)

    def get_duration_seconds(self) -> float:
        return len(self._buffer) / self.config.bytes_per_second

    def get_sample_count(self) -> int:
        return len(self._buffer) // self.config.bytes_per_sample

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0


class FrameBuffer:
    def __init__(self, frame_size: int = 512):
        self.frame_size = frame_size
        self._buffer: bytearray = bytearray()

    def append(self, data: bytes) -> None:
        self._buffer.extend(data)

    def extract_frames(self) -> list[bytes]:
        frames = []
        while len(self._buffer) >= self.frame_size:
            frame = bytes(self._buffer[: self.frame_size])
            frames.append(frame)
            self._buffer = self._buffer[self.frame_size :]
        return frames

    def has_complete_frame(self) -> bool:
        return len(self._buffer) >= self.frame_size

    def get_remaining(self) -> bytes:
        return bytes(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
