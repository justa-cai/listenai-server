"""Tests for audio processor."""

import pytest
import numpy as np
from src.audio_processor import AudioProcessor


@pytest.fixture
def processor():
    """Create an audio processor instance."""
    return AudioProcessor()


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_to_pcm16_float32(self, processor):
        """Test converting float32 audio to PCM16."""
        audio = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
        result = processor.to_pcm16(audio)

        assert isinstance(result, bytes)
        assert len(result) == audio.shape[0] * 2  # 2 bytes per sample

    def test_to_pcm16_int16(self, processor):
        """Test converting int16 audio to PCM16."""
        audio = np.array([0, 16384, 32767, -16384, -32768], dtype=np.int16)
        result = processor.to_pcm16(audio)

        assert isinstance(result, bytes)
        assert len(result) == audio.shape[0] * 2

    def test_to_pcm16_clipping(self, processor):
        """Test that float values are clipped to [-1, 1]."""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        result = processor.to_pcm16(audio)

        # Should be clipped to max int16 values
        decoded = processor.from_pcm16(result)
        assert decoded[0] <= 1.0
        assert decoded[1] >= -1.0

    def test_from_pcm16(self, processor):
        """Test converting PCM16 bytes to float32."""
        audio = np.array([0.5, -0.5], dtype=np.float32)
        pcm_bytes = processor.to_pcm16(audio)
        result = processor.from_pcm16(pcm_bytes)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 2
        # Allow some precision loss
        assert abs(result[0] - 0.5) < 0.01
        assert abs(result[1] - (-0.5)) < 0.01

    def test_roundtrip_conversion(self, processor):
        """Test roundtrip conversion preserves audio data."""
        original = np.random.randn(1000).astype(np.float32)
        # Normalize to prevent clipping
        original = original / np.max(np.abs(original)) * 0.9

        pcm_bytes = processor.to_pcm16(original)
        recovered = processor.from_pcm16(pcm_bytes)

        # Check that recovered signal is close to original
        correlation = np.corrcoef(original, recovered)[0, 1]
        assert correlation > 0.999  # Should be very close

    def test_normalize_audio(self, processor):
        """Test audio normalization."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        normalized = processor.normalize_audio(audio, target_level=0.5)

        peak = np.max(np.abs(normalized))
        assert abs(peak - 0.5) < 0.01

    def test_get_audio_info(self, processor):
        """Test getting audio information."""
        audio = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
        sample_rate = 24000

        info = processor.get_audio_info(audio, sample_rate)

        assert info["samples"] == 24000
        assert info["duration"] == 1.0
        assert info["sample_rate"] == 24000
        assert info["channels"] == 1

    def test_chunk_audio(self, processor):
        """Test chunking audio into segments."""
        audio = np.random.randn(10000).astype(np.float32)
        chunk_size = 2000
        sample_rate = 24000

        chunks = processor.chunk_audio(audio, chunk_size, sample_rate)

        assert len(chunks) == 5  # 10000 / 2000 = 5
        assert all(len(chunk) <= chunk_size for chunk in chunks)

    def test_calculate_rms(self, processor):
        """Test RMS calculation."""
        audio = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        rms = processor.calculate_rms(audio)

        # RMS of [1, -1, 1, -1] should be 1.0
        assert abs(rms - 1.0) < 0.01

    def test_detect_silence(self, processor):
        """Test silence detection."""
        # Create audio with silent regions
        audio = np.concatenate([
            np.ones(1000) * 0.5,  # Non-silent
            np.zeros(3000),  # Silent (0.125s at 24kHz)
            np.ones(1000) * 0.5,  # Non-silent
        ])

        silent_regions = processor.detect_silence(
            audio,
            threshold=0.01,
            min_duration=0.1,
            sample_rate=24000
        )

        assert len(silent_regions) == 1
        assert silent_regions[0][0] == 1000
        assert silent_regions[0][1] == 4000
