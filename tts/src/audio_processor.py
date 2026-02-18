"""Audio processing utilities."""

import numpy as np
from typing import Tuple


class AudioProcessor:
    """Processor for audio data conversion and manipulation."""

    def __init__(self):
        """Initialize the audio processor."""
        pass

    def to_pcm16(self, audio: np.ndarray) -> bytes:
        """
        Convert audio array to PCM 16-bit bytes.

        Args:
            audio: Audio data as numpy array

        Returns:
            PCM 16-bit encoded bytes
        """
        # Handle different input dtypes
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Normalize float audio to [-1, 1] range
            audio = np.clip(audio, -1.0, 1.0)
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
        elif audio.dtype == np.int16:
            audio_int16 = audio
        elif audio.dtype == np.int32:
            # Convert int32 to int16
            audio_int16 = (audio.astype(np.float32) / (2**16)).astype(np.int16)
        elif audio.dtype == np.uint8:
            # Convert uint8 to int16
            audio_int16 = ((audio.astype(np.float32) - 128) * 256).astype(np.int16)
        else:
            raise ValueError(f"Unsupported audio dtype: {audio.dtype}")

        return audio_int16.tobytes()

    def from_pcm16(self, data: bytes) -> np.ndarray:
        """
        Convert PCM 16-bit bytes to audio array.

        Args:
            data: PCM 16-bit encoded bytes

        Returns:
            Audio data as numpy array (float32)
        """
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32767.0

    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """
        Normalize audio to target peak level.

        Args:
            audio: Input audio array
            target_level: Target peak level (0.0 to 1.0)

        Returns:
            Normalized audio array
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * target_level

        return audio

    def get_audio_info(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Get information about audio data.

        Args:
            audio: Audio array
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary with audio information
        """
        duration = len(audio) / sample_rate if sample_rate > 0 else 0

        return {
            "samples": len(audio),
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": 1 if audio.ndim == 1 else audio.shape[1],
            "dtype": str(audio.dtype),
        }

    def chunk_audio(
        self,
        audio: np.ndarray,
        chunk_size: int,
        sample_rate: int
    ) -> list[np.ndarray]:
        """
        Split audio into chunks.

        Args:
            audio: Input audio array
            chunk_size: Number of samples per chunk
            sample_rate: Sample rate in Hz

        Returns:
            List of audio chunks
        """
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def calculate_rms(self, audio: np.ndarray) -> float:
        """
        Calculate RMS (root mean square) amplitude of audio.

        Args:
            audio: Input audio array

        Returns:
            RMS value
        """
        return float(np.sqrt(np.mean(audio ** 2)))

    def detect_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        min_duration: float = 0.1,
        sample_rate: int = 24000
    ) -> list[Tuple[int, int]]:
        """
        Detect silent regions in audio.

        Args:
            audio: Input audio array
            threshold: Amplitude threshold for silence
            min_duration: Minimum silence duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            List of (start, end) sample indices for silent regions
        """
        is_silent = np.abs(audio) < threshold
        min_samples = int(min_duration * sample_rate)

        silent_regions = []
        start = None

        for i, silent in enumerate(is_silent):
            if silent and start is None:
                start = i
            elif not silent and start is not None:
                if i - start >= min_samples:
                    silent_regions.append((start, i))
                start = None

        # Handle trailing silence
        if start is not None and len(audio) - start >= min_samples:
            silent_regions.append((start, len(audio)))

        return silent_regions
