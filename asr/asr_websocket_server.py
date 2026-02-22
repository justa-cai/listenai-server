#!/usr/bin/env python3
"""
ASR WebSocket Server - Real-time Speech Recognition Service
"""

import asyncio
import json
import logging
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Optional, Deque, Tuple

import numpy as np
import soundfile as sf
import torch
import websockets
from aiohttp import web

from model import FunASRNano


# ============================================================================
# Configuration
# ============================================================================

# VAD Configuration
VAD_HOP_SIZE = 256
VAD_THRESHOLD = 0.5
VAD_SPEECH_FRAMES = 3  # Need N consecutive speech frames to enter speech state (prevents false triggers)
VAD_SILENCE_FRAMES = 5  # Need N consecutive silence frames to exit speech state (hysteresis)

# ASR Configuration
ASR_MODEL_DIR = "FunAudioLLM/Fun-ASR-Nano-2512"
ASR_SAMPLE_RATE = 16000  # Hz
BUFFER_MAX_DURATION = 2.0  # seconds
ASR_LANGUAGE = "中文"  # Language for ASR: 中文, 英文, etc.
ASR_TEMP_DIR = "tmp"  # Directory for temporary wav files for debugging
ASR_KEEP_TEMP_FILES = True  # Keep temp wav files for debugging (set to False to delete)

# Energy Filter Configuration
ENERGY_THRESHOLD = 0.01  # Energy threshold for filtering low-energy speech segments (noise)
ASR_MIN_TEXT_LENGTH = 2  # Minimum text length to send result (filters empty/invalid ASR results)
VAD_SPEECH_RATIO_THRESHOLD = 0.3  # Minimum speech ratio (speech frames / total frames) for valid audio

# Noise Suppression Configuration
NS_ENABLED = False # Enable/disable noise suppression (RNNoise)
NS_SAMPLE_RATE = 16000  # RNNoise sample rate (must match ASR sample rate)

# WebSocket Configuration
WS_HOST = "0.0.0.0"
WS_PORT = 9200
WS_TIMEOUT = 60  # seconds
WS_PING_INTERVAL = 20  # seconds

# HTTP Configuration
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 9201
HTML_FILE = "websocket_asr_client.html"

# Logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(
    logging.INFO
)  # Default to INFO level (set to DEBUG for verbose logging)


# ============================================================================
# Audio Buffer
# ============================================================================


# ============================================================================
# Frame Buffer - Cross-message audio buffering for VAD frames
# ============================================================================


class FrameBuffer:
    """
    Buffer for accumulating audio data across WebSocket messages.
    Ensures complete VAD frames are processed without data loss.

    When a message size is not a multiple of the VAD frame size,
    the remaining bytes are stored and combined with the next message.

    Example with 20ms interval (640 bytes, 1.25 frames):
    - Message 1: 640 bytes → Extract 1 frame (512 bytes), keep 128 bytes
    - Message 2: 640 bytes → 128 + 640 = 768 bytes → Extract 1 frame, keep 256 bytes
    - Message 3: 640 bytes → 256 + 640 = 896 bytes → Extract 1 frame, keep 384 bytes
    """

    def __init__(self, frame_size: int = VAD_HOP_SIZE * 2):
        """
        Args:
            frame_size: Size of one VAD frame in bytes (default: 512 bytes for 256 samples * 2 bytes/sample)
        """
        self.frame_size = frame_size
        self.buffer: bytes = b""  # Accumulated audio data

    def add(self, audio_data: bytes) -> list[bytes]:
        """
        Add new audio data and extract complete VAD frames.

        Args:
            audio_data: Raw audio bytes from WebSocket message

        Returns:
            List of complete VAD frames (each exactly frame_size bytes)
        """
        # Combine with existing buffer
        self.buffer += audio_data

        frames = []

        # Extract complete frames
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[: self.frame_size]
            frames.append(frame)
            self.buffer = self.buffer[self.frame_size :]

        return frames

    def has_remaining(self) -> bool:
        """Check if there's remaining data in the buffer."""
        return len(self.buffer) > 0

    def get_remaining_size(self) -> int:
        """Get size of remaining data in bytes."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = b""


class AudioBuffer:
    """
    Audio data buffer for managing incoming audio stream.
    Maximum buffer size: 2 seconds at 16kHz = 32000 samples
    """

    def __init__(
        self,
        sample_rate: int = ASR_SAMPLE_RATE,
        max_duration: float = BUFFER_MAX_DURATION,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer: Deque[np.ndarray] = deque()
        self.total_samples = 0
        self.speech_start_time: Optional[float] = None
        self.speech_end_time: Optional[float] = None

    def add(self, audio_data: bytes) -> int:
        """Add audio data to buffer. Returns number of samples added."""
        # Convert bytes to int16 numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        samples_added = len(audio)

        # Add to buffer
        self.buffer.append(audio)
        self.total_samples += samples_added

        # Drop oldest data if buffer exceeds max size
        while self.total_samples > self.max_samples:
            oldest = self.buffer.popleft()
            self.total_samples -= len(oldest)

        return samples_added

    def get_vad_frames(self, hop_size: int = VAD_HOP_SIZE) -> list[np.ndarray]:
        """Extract VAD frames from buffer."""
        audio = self.get_audio()
        frames = []
        for i in range(0, len(audio), hop_size):
            frame = audio[i : i + hop_size]
            if len(frame) < hop_size:
                # Pad with zeros
                frame = np.pad(frame, (0, hop_size - len(frame)))
            frames.append(frame)
        return frames

    def get_audio(self) -> np.ndarray:
        """Get all audio in buffer as a single array."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.buffer))

    def get_duration(self) -> float:
        """Get duration of audio in buffer in seconds."""
        return self.total_samples / self.sample_rate

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.total_samples = 0
        self.speech_start_time = None
        self.speech_end_time = None

    def mark_speech_start(self) -> None:
        """Mark speech start time."""
        self.speech_start_time = time.time()

    def mark_speech_end(self) -> None:
        """Mark speech end time."""
        self.speech_end_time = time.time()


class SpeechBuffer:
    """
    Speech segment buffer - stores audio from speech start to speech end.
    Used for ASR recognition after VAD detects speech end.
    """

    def __init__(self, sample_rate: int = ASR_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.buffer: list[np.ndarray] = []
        self.total_samples = 0
        self.is_recording = False
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start recording speech segment."""
        self.buffer.clear()
        self.total_samples = 0
        self.is_recording = True
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop recording speech segment."""
        self.is_recording = False

    def add(self, audio_data: np.ndarray) -> None:
        """Add audio data to speech buffer (only when recording)."""
        if self.is_recording:
            self.buffer.append(audio_data)
            self.total_samples += len(audio_data)

    def get_audio(self) -> np.ndarray:
        """Get all audio in speech buffer."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.buffer)

    def get_duration(self) -> float:
        """Get duration of speech in seconds."""
        return self.total_samples / self.sample_rate

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.total_samples = 0
        self.is_recording = False
        self.start_time = None

    def get_rms_energy(self) -> float:
        """
        Calculate RMS (Root Mean Square) energy of the audio buffer.

        Returns:
            RMS energy value (0.0 to 1.0 for normalized audio)
        """
        if self.total_samples == 0:
            return 0.0

        audio = self.get_audio()
        return np.sqrt(np.mean(audio ** 2))


# ============================================================================
# Noise Suppression (NS) Processor - RNNoise
# ============================================================================

try:
    from pyrnnoise import RNNoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False
    logger.warning(
        "pyrnnoise not installed. Noise suppression disabled. "
        "Install with: pip install pyrnnoise"
    )


class NSProcessor:
    """
    Noise Suppression processor using RNNoise (Recurrent Neural Network Noise Reduction).

    RNNoise is a lightweight deep learning based noise suppression algorithm
    developed by Mozilla/Xiph. It uses a GRU-based neural network to predict
    gain masks for each frequency band, effectively suppressing background noise
    while preserving speech quality.

    Features:
    - Very low latency (<10ms)
    - Lightweight model (~200KB)
    - Real-time processing capable
    - Effective against various noise types (fan, keyboard, traffic, etc.)

    Note: RNNoise native sample rate is 48kHz with fixed frame size of 480 samples.
    This class handles buffering and resampling to match VAD frame sizes.
    """

    # RNNoise native configuration
    RNNOISE_SAMPLE_RATE = 48000  # RNNoise native sample rate
    RNNOISE_FRAME_SIZE = 480  # 10ms at 48kHz

    def __init__(self, sample_rate: int = NS_SAMPLE_RATE, enabled: bool = NS_ENABLED):
        """
        Initialize NS Processor.

        Args:
            sample_rate: Sample rate for processing (default: 16000 Hz)
            enabled: Whether noise suppression is enabled
        """
        self.enabled = enabled and RNNOISE_AVAILABLE
        self.sample_rate = sample_rate
        self.resample_needed = sample_rate != self.RNNOISE_SAMPLE_RATE

        # Buffer for accumulating audio at 48kHz
        self._buffer_48k: list[np.ndarray] = []
        self._buffer_size_48k = 0

        # Buffer for output at original sample rate
        self._output_buffer: list[np.ndarray] = []
        self._output_buffer_size = 0

        if self.enabled:
            try:
                # Initialize RNNoise at its native 48kHz
                self.denoiser = RNNoise(sample_rate=self.RNNOISE_SAMPLE_RATE)
                self.frame_size = int(0.01 * sample_rate)  # 10ms frame size at input sample rate
                logger.info(
                    f"RNNoise initialized: native_rate={self.RNNOISE_SAMPLE_RATE}Hz, "
                    f"input_rate={sample_rate}Hz, rnnoise_frame={self.RNNOISE_FRAME_SIZE}samples, "
                    f"resample={'enabled' if self.resample_needed else 'not needed'}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize RNNoise: {e}")
                self.enabled = False
        else:
            if not RNNOISE_AVAILABLE:
                logger.info("Noise suppression disabled (pyrnnoise not available)")
            else:
                logger.info("Noise suppression disabled (NS_ENABLED=False)")

    def _resample_to_rnnoise(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from input sample rate to RNNoise's 48kHz."""
        if not self.resample_needed:
            return audio

        # Calculate target length
        ratio = self.RNNOISE_SAMPLE_RATE / self.sample_rate
        target_length = int(len(audio) * ratio)

        # Simple linear interpolation resampling
        # For better quality, consider using scipy.signal.resample or librosa.resample
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _resample_from_rnnoise(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Resample audio from RNNoise's 48kHz back to input sample rate."""
        if not self.resample_needed:
            return audio[:target_length]

        # Calculate source indices
        ratio = self.sample_rate / self.RNNOISE_SAMPLE_RATE
        indices = np.linspace(0, len(audio) - 1, target_length)

        # Simple linear interpolation resampling
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _process_rnnoise_frames(self) -> None:
        """Process all complete RNNoise frames in the buffer."""
        while self._buffer_size_48k >= self.RNNOISE_FRAME_SIZE:
            # Extract one RNNoise frame (480 samples at 48kHz)
            frame_parts = []
            samples_needed = self.RNNOISE_FRAME_SIZE
            remaining_samples = samples_needed

            # Collect samples from buffer
            while remaining_samples > 0 and self._buffer_48k:
                chunk = self._buffer_48k[0]
                if len(chunk) <= remaining_samples:
                    # Take the whole chunk
                    frame_parts.append(chunk)
                    remaining_samples -= len(chunk)
                    self._buffer_48k.pop(0)
                else:
                    # Take part of the chunk
                    frame_parts.append(chunk[:remaining_samples])
                    self._buffer_48k[0] = chunk[remaining_samples:]
                    remaining_samples = 0

            # Concatenate frame parts
            frame_48k = np.concatenate(frame_parts) if len(frame_parts) > 1 else frame_parts[0]
            self._buffer_size_48k -= self.RNNOISE_FRAME_SIZE

            # Process with RNNoise
            try:
                # Convert to int16 and reshape for denoise_chunk [1, 480]
                frame_int16 = (frame_48k * 32768).astype(np.int16)
                frame_chunk = frame_int16.reshape(1, -1)

                results = list(self.denoiser.denoise_chunk(frame_chunk, partial=False))

                if results and results[0] and len(results[0]) >= 2:
                    _, denoised_int16 = results[0]
                    if denoised_int16 is not None:
                        if denoised_int16.ndim == 2:
                            denoised_int16 = denoised_int16[0]
                        denoised_float32 = denoised_int16.astype(np.float32) / 32768.0
                        self._output_buffer.append(denoised_float32)
                        self._output_buffer_size += len(denoised_float32)
            except Exception as e:
                logger.error(f"RNNoise frame processing error: {e}")

    def _get_output(self, num_samples: int) -> np.ndarray:
        """Get specified number of samples from output buffer."""
        if not self._output_buffer:
            return None

        # Collect samples from output buffer
        output_parts = []
        samples_collected = 0
        remaining = num_samples

        while remaining > 0 and self._output_buffer:
            chunk = self._output_buffer[0]
            if len(chunk) <= remaining:
                output_parts.append(chunk)
                samples_collected += len(chunk)
                remaining -= len(chunk)
                self._output_buffer.pop(0)
                self._output_buffer_size -= len(chunk)
            else:
                output_parts.append(chunk[:remaining])
                samples_collected += remaining
                self._output_buffer[0] = chunk[remaining:]
                self._output_buffer_size -= remaining
                remaining = 0

        if samples_collected < num_samples:
            # Not enough output, pad with zeros
            output = np.concatenate(output_parts) if output_parts else np.array([], dtype=np.float32)
            if len(output) < num_samples:
                output = np.pad(output, (0, num_samples - len(output)), mode='constant')
            return output

        return np.concatenate(output_parts) if len(output_parts) > 1 else output_parts[0]
        """Resample audio from RNNoise's 48kHz back to input sample rate."""
        if not self.resample_needed:
            return audio[:target_length]

        # Calculate source indices
        ratio = self.sample_rate / self.RNNOISE_SAMPLE_RATE
        indices = np.linspace(0, len(audio) - 1, target_length)

        # Simple linear interpolation resampling
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def process_frame(self, audio: np.ndarray) -> np.ndarray:
        """
        Process a single audio frame through RNNoise with buffering.

        This method handles frame size mismatches between VAD (variable) and
        RNNoise (fixed 480 samples @ 48kHz) using an internal buffer.

        Args:
            audio: Audio data as float32 numpy array (normalized to [-1, 1])

        Returns:
            Denoised audio as float32 numpy array (same length as input)
        """
        if not self.enabled:
            return audio

        try:
            original_length = len(audio)

            # Resample to RNNoise's native 48kHz if needed
            audio_48k = self._resample_to_rnnoise(audio) if self.resample_needed else audio

            # Add to buffer
            self._buffer_48k.append(audio_48k)
            self._buffer_size_48k += len(audio_48k)

            # Process all complete RNNoise frames
            self._process_rnnoise_frames()

            # Get output (resampled back to original sample rate)
            if self._output_buffer:
                if self.resample_needed:
                    # Calculate how many 48kHz samples we need for the output
                    output_48k_length = int(original_length * self.RNNOISE_SAMPLE_RATE / self.sample_rate)
                    output_48k = self._get_output(output_48k_length)
                    if output_48k is not None:
                        return self._resample_from_rnnoise(output_48k, original_length)
                else:
                    output = self._get_output(original_length)
                    if output is not None:
                        return output

            # Not enough output yet, return original audio
            return audio

        except Exception as e:
            logger.error(f"RNNoise processing error: {e}")
            return audio  # Return original audio on error

    def reset(self) -> None:
        """Reset internal buffers."""
        self._buffer_48k.clear()
        self._buffer_size_48k = 0
        self._output_buffer.clear()
        self._output_buffer_size = 0

    def is_enabled(self) -> bool:
        """Check if noise suppression is enabled and available."""
        return self.enabled


# ============================================================================
# VAD Processor
# ============================================================================

# Import TenVad from local path
import sys
import os

ten_vad_path = os.path.join(os.path.dirname(__file__), "../vad/ten-vad/include")
sys.path.insert(0, ten_vad_path)
from ten_vad import TenVad


class VADProcessor:
    """
    Voice Activity Detection using TenVad with dual hysteresis.

    Dual hysteresis mechanism:
    - Enter speech state only after N consecutive speech frames (prevents false triggers)
    - Exit speech state only after N consecutive silence frames (prevents early exit)

    This prevents sudden noises (coughs, clicks) from triggering speech detection,
    while also preventing brief pauses from ending speech segments prematurely.
    """

    def __init__(
        self,
        hop_size: int = VAD_HOP_SIZE,
        threshold: float = VAD_THRESHOLD,
        speech_frames: int = VAD_SPEECH_FRAMES,
        silence_frames: int = VAD_SILENCE_FRAMES,
    ):
        self.hop_size = hop_size
        self.threshold = threshold
        self.speech_frames = speech_frames  # Number of consecutive speech frames to enter speech
        self.silence_frames = silence_frames  # Number of consecutive silence frames to exit speech

        # Initialize TenVad
        try:
            self.ten_vad = TenVad(hop_size=hop_size, threshold=threshold)
            logger.info(
                f"TenVad initialized with hop_size={hop_size}, threshold={threshold}, "
                f"speech_frames={speech_frames}, silence_frames={silence_frames}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize TenVad: {e}")
            raise

        # VAD state machine
        self.is_speeching = False  # Current state
        self.consecutive_speech_count = 0  # Consecutive speech frame counter (for entering speech)
        self.consecutive_silence_count = 0  # Consecutive silence frame counter (for exiting speech)
        self.frame_count = 0  # Total frames processed

    def process_frame(self, frame_audio: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Process a single VAD frame using TenVad.

        Args:
            frame_audio: Audio data for one frame (float32, should be hop_size samples)

        Returns:
            (is_speeching, probability) - is_speeching indicates the current state
        """
        self.frame_count += 1

        # Convert float32 audio to int16 for TenVad
        audio_int16 = (frame_audio * 32768).astype(np.int16)

        # Ensure frame has exactly hop_size samples
        if len(audio_int16) < self.hop_size:
            audio_int16 = np.pad(
                audio_int16, (0, self.hop_size - len(audio_int16)), mode="constant"
            )

        try:
            # Process with TenVad
            probability, speech_flag = self.ten_vad.process(audio_int16)
        except Exception as e:
            logger.error(f"TenVad process error at frame {self.frame_count}: {e}")
            return self.is_speeching, None

        # speech_flag: 0 = non-speech, 1 = speech
        is_speech_frame = speech_flag == 1

        # Dual hysteresis state machine
        if is_speech_frame:
            # Speech frame detected
            if not self.is_speeching:
                # Not in speech state: increment speech counter
                self.consecutive_speech_count += 1

                # Check if we should enter speech state
                if self.consecutive_speech_count >= self.speech_frames:
                    self.is_speeching = True
                    self.consecutive_speech_count = 0
                    self.consecutive_silence_count = 0
                    logger.info(
                        f"Speech START at frame {self.frame_count} (after {self.speech_frames} consecutive speech frames), prob: {probability:.3f}"
                    )
            else:
                # Already in speech state: continue, reset silence counter
                self.consecutive_silence_count = 0
        else:
            # Silence frame detected
            if self.is_speeching:
                # In speech state: increment silence counter
                self.consecutive_silence_count += 1
                self.consecutive_speech_count = 0  # Reset speech counter

                # Check if we should exit speech state
                if self.consecutive_silence_count >= self.silence_frames:
                    self.is_speeching = False
                    self.consecutive_silence_count = 0
                    logger.info(
                        f"Speech END at frame {self.frame_count} (after {self.silence_frames} consecutive silence frames)"
                    )
            else:
                # Already in silence state: reset speech counter
                self.consecutive_speech_count = 0

        # Debug logging for every frame (use DEBUG to reduce log spam)
        logger.debug(
            f"VAD Frame {self.frame_count}: prob={probability:.3f}, flag={speech_flag}, "
            f"is_speeching={self.is_speeching}, "
            f"speech_count={self.consecutive_speech_count}/{self.speech_frames}, "
            f"silence_count={self.consecutive_silence_count}/{self.silence_frames}"
        )

        return self.is_speeching, probability

    # Deprecated: kept for backward compatibility
    def process(self, audio_buffer: AudioBuffer) -> Tuple[bool, Optional[float]]:
        """Process audio buffer (deprecated, use process_frame instead)."""
        frames = audio_buffer.get_vad_frames(self.hop_size)
        if not frames:
            return self.is_speeching, None

        probability = 0.0
        for frame in frames:
            is_speech, probability = self.process_frame(frame)

        return self.is_speeching, probability

    def reset(self) -> None:
        """Reset VAD state."""
        self.is_speeching = False
        self.consecutive_speech_count = 0
        self.consecutive_silence_count = 0
        self.frame_count = 0

    def calculate_speech_ratio(self, audio: np.ndarray) -> float:
        """
        Calculate the ratio of speech frames to total frames in audio.

        This helps distinguish between valid speech (high speech ratio) and
        sudden noises (low speech ratio despite high energy).

        Args:
            audio: Audio data (float32 numpy array)

        Returns:
            Speech ratio (0.0 to 1.0), where 1.0 means all frames are speech
        """
        if len(audio) == 0:
            return 0.0

        # Save current state
        saved_is_speeching = self.is_speeching
        saved_silence_count = self.consecutive_silence_count
        saved_frame_count = self.frame_count

        # Split audio into frames and count speech frames
        total_frames = 0
        speech_frames = 0

        for i in range(0, len(audio), self.hop_size):
            frame = audio[i : i + self.hop_size]
            if len(frame) < self.hop_size:
                # Pad last frame if needed
                frame = np.pad(frame, (0, self.hop_size - len(frame)))

            # Convert float32 to int16 for TenVad
            frame_int16 = (frame * 32768.0).astype(np.int16)
            if len(frame_int16) < self.hop_size:
                frame_int16 = np.pad(
                    frame_int16, (0, self.hop_size - len(frame_int16)), mode="constant"
                )

            # Use TenVad to check if this frame is speech
            probability, speech_flag = self.ten_vad.process(frame_int16)
            total_frames += 1
            if speech_flag == 1:  # 1 = speech, 0 = non-speech
                speech_frames += 1

        # Restore state
        self.is_speeching = saved_is_speeching
        self.consecutive_silence_count = saved_silence_count
        self.frame_count = saved_frame_count

        if total_frames == 0:
            return 0.0

        return speech_frames / total_frames


# ============================================================================
# ASR Processor
# ============================================================================


class ASRProcessor:
    """
    Automatic Speech Recognition processor using FunASR-Nano model.
    Supports both streaming and segment-based recognition.
    """

    def __init__(
        self,
        model_dir: str = ASR_MODEL_DIR,
        sample_rate: int = ASR_SAMPLE_RATE,
        device: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.segment_counter = 0

        # Auto-select device
        if device is None:
            device = (
                "cuda:0"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.device = device
        print(f"Loading ASR model from {model_dir} on {device}...")
        logger.info(f"Loading ASR model from {model_dir} on {device}")

        # Load model
        self.model, self.kwargs = FunASRNano.from_pretrained(
            model=model_dir, device=device
        )
        self.model.eval()

        print("ASR model loaded successfully")
        logger.info("ASR model loaded successfully")

    def _process_audio_data(
        self, audio: np.ndarray, segment_id: int
    ) -> Optional[Tuple[str, int]]:
        """
        Internal method to process audio data and save to wav file.
        Returns (text, segment_id) or None.
        """
        # Create tmp directory if it doesn't exist
        temp_dir = Path(ASR_TEMP_DIR)
        temp_dir.mkdir(exist_ok=True)

        # Generate wav filename with timestamp and segment_id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_wav_path = temp_dir / f"segment_{timestamp}_id{segment_id}.wav"

        try:
            # Write audio data to wav file (16kHz, 16bit PCM)
            sf.write(str(temp_wav_path), audio, self.sample_rate, subtype="PCM_16")

            duration = len(audio) / self.sample_rate
            logger.info(
                f"Created wav file: {temp_wav_path}, duration: {duration:.2f}s, samples: {len(audio)}"
            )

            # Run inference with wav file path (matching demo2.py exactly)
            # Returns: (results, meta_data) where results is a list of dicts
            res = self.model.inference(data_in=[str(temp_wav_path)], **self.kwargs)

            # Extract text from result
            # Format: res[0] is a dict with 'text' key, or res[0][0] might be text directly
            result = res[0]
            if isinstance(result, dict):
                text = result.get("text", "")
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    text = result[0].get("text", "")
                else:
                    text = str(result[0])
            else:
                text = str(result)

            logger.info(f"Segment ASR result (ID={segment_id}): {text}")

            # Optionally delete temp file (set ASR_KEEP_TEMP_FILES to False to auto-delete)
            if not ASR_KEEP_TEMP_FILES:
                temp_wav_path.unlink()
                logger.info(f"Deleted wav file: {temp_wav_path}")
            else:
                logger.info(f"Kept wav file for analysis: {temp_wav_path}")

            return text, segment_id

        except Exception as e:
            logger.error(f"Error in segment ASR: {e}\n{traceback.format_exc()}")
            return None

    def process_segment_from_buffer(
        self, speech_buffer: SpeechBuffer
    ) -> Optional[Tuple[str, int]]:
        """
        Process complete speech segment from SpeechBuffer for final recognition.
        Returns (text, segment_id) or None.
        """
        if speech_buffer.is_empty():
            return None

        audio = speech_buffer.get_audio()
        if len(audio) < self.sample_rate * 0.3:  # Minimum 0.3 seconds
            return None

        self.segment_counter += 1
        segment_id = self.segment_counter

        return self._process_audio_data(audio, segment_id)

    def process_segment(self, audio_buffer: AudioBuffer) -> Optional[Tuple[str, int]]:
        """
        Process complete speech segment for final recognition.
        Uses temporary wav file in local tmp/ directory (same as demo2.py approach).
        Returns (text, segment_id) or None.
        """
        audio = audio_buffer.get_audio()
        if len(audio) < self.sample_rate * 0.3:  # Minimum 0.3 seconds
            return None

        self.segment_counter += 1
        segment_id = self.segment_counter

        return self._process_audio_data(audio, segment_id)

    @staticmethod
    def is_valid_asr_result(text: str) -> bool:
        """
        Validate ASR result to filter out invalid/non-speech audio.

        Checks:
        - Text not empty after stripping
        - Minimum length threshold
        - Not only punctuation/symbols

        Args:
            text: ASR recognition result text

        Returns:
            True if result is valid speech, False if should be filtered
        """
        if not text:
            return False

        # Remove whitespace
        text_stripped = text.strip()

        if not text_stripped:
            return False

        # Check minimum length (filters single char results)
        if len(text_stripped) < ASR_MIN_TEXT_LENGTH:
            return False

        # Check if result contains at least some Chinese characters or alphanumeric
        # This filters pure punctuation results like "，。" or "?!"
        import re
        has_valid_chars = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9]', text_stripped))
        if not has_valid_chars:
            return False

        return True


# ============================================================================
# ASR WebSocket Service
# ============================================================================


class ASRWebSocketService:
    """
    Main WebSocket service for real-time ASR.
    """

    def __init__(
        self,
        host: str = WS_HOST,
        port: int = WS_PORT,
        timeout: int = WS_TIMEOUT,
        ping_interval: int = WS_PING_INTERVAL,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.ping_interval = ping_interval

        # Load ASR model (synchronous, will block until loaded)
        logger.info("Initializing ASR service...")
        self.asr_processor = ASRProcessor()
        logger.info("ASR service initialized")

    async def handle_client(self, websocket):
        """Handle individual WebSocket client connection."""
        client_id = id(websocket)
        logger.info(f"[Client {client_id}] Connected from {websocket.remote_address}")

        # Initialize client-specific buffers and processors
        frame_buffer = FrameBuffer()  # Cross-message frame buffering (NO DATA LOSS!)
        speech_buffer = SpeechBuffer()  # For storing speech segments
        ns_processor = NSProcessor()  # Noise suppression (RNNoise)
        vad_processor = VADProcessor()

        try:
            async for message in websocket:
                # Check if message is binary (audio data)
                if isinstance(message, bytes):
                    # Step 1: Apply Noise Suppression (NS) on raw audio data first
                    if ns_processor.is_enabled():
                        # Convert bytes to float32
                        audio_float32 = (
                            np.frombuffer(message, dtype=np.int16).astype(
                                np.float32
                            )
                            / 32768.0
                        )
                        # Process through NS (handles buffering internally)
                        denoised_float32 = ns_processor.process_frame(audio_float32)
                        # Convert back to bytes for FrameBuffer
                        message = (denoised_float32 * 32768).astype(np.int16).tobytes()

                    # Step 2: Add to frame buffer and extract complete VAD frames
                    frames = frame_buffer.add(message)

                    # Debug logging (use DEBUG level to reduce spam)
                    logger.debug(
                        f"[Client {client_id}] Received {len(message)} bytes, extracted {len(frames)} frame(s), {frame_buffer.get_remaining_size()} bytes remaining"
                    )

                    # Step 3: Process each complete VAD frame
                    for chunk_bytes in frames:
                        # Convert bytes to float32 audio for VAD and speech_buffer
                        chunk_float32 = (
                            np.frombuffer(chunk_bytes, dtype=np.int16).astype(
                                np.float32
                            )
                            / 32768.0
                        )

                        # Process this single VAD frame (this may change the state)
                        was_speeching = vad_processor.is_speeching
                        is_speeching, _ = vad_processor.process_frame(chunk_float32)

                        # State machine: handle speech segment boundaries
                        if not was_speeching and is_speeching:
                            # Speech segment STARTED
                            speech_buffer.start()
                            speech_buffer.add(chunk_float32)
                            logger.info(f"[Client {client_id}] Speech segment started")
                            # Send VAD event to client
                            await self.send_vad_event(websocket, speech_started=True)
                        elif was_speeching and is_speeching:
                            # Still in speech - continue adding
                            speech_buffer.add(chunk_float32)
                        elif was_speeching and not is_speeching:
                            # Speech segment ENDED - add final frame and process
                            speech_buffer.add(chunk_float32)
                            speech_buffer.stop()

                            duration = speech_buffer.get_duration()
                            samples = speech_buffer.total_samples

                            # Send VAD event to client
                            await self.send_vad_event(websocket, speech_started=False, duration=duration, samples=samples)

                            # Calculate energy for filtering
                            energy = speech_buffer.get_rms_energy()
                            logger.info(
                                f"[Client {client_id}] Speech segment ended - "
                                f"duration: {duration:.2f}s, samples: {samples}, "
                                f"energy: {energy:.6f}"
                            )

                            # Filter by energy threshold
                            if energy < ENERGY_THRESHOLD:
                                logger.info(
                                    f"[Client {client_id}] Speech segment energy too low "
                                    f"({energy:.6f} < {ENERGY_THRESHOLD}), skipping ASR"
                                )
                                speech_buffer.clear()
                                continue

                            # Calculate speech ratio for filtering sudden noises
                            speech_audio = speech_buffer.get_audio()
                            speech_ratio = vad_processor.calculate_speech_ratio(speech_audio)
                            logger.info(
                                f"[Client {client_id}] Speech ratio: {speech_ratio:.2%}"
                            )

                            # Filter by speech ratio threshold
                            if speech_ratio < VAD_SPEECH_RATIO_THRESHOLD:
                                logger.info(
                                    f"[Client {client_id}] Speech ratio too low "
                                    f"({speech_ratio:.2%} < {VAD_SPEECH_RATIO_THRESHOLD:.0%}), "
                                    f"likely noise, skipping ASR"
                                )
                                speech_buffer.clear()
                                continue

                            if duration >= 0.3:  # Minimum 0.3 seconds
                                logger.info(
                                    f"[Client {client_id}] Running segment recognition..."
                                )
                                result = self.asr_processor.process_segment_from_buffer(
                                    speech_buffer
                                )
                                if result:
                                    text, segment_id = result
                                    logger.info(
                                        f"[Client {client_id}] Segment recognition result: {text}"
                                    )

                                    # Validate ASR result to filter invalid/non-speech audio
                                    if not self.asr_processor.is_valid_asr_result(text):
                                        logger.info(
                                            f"[Client {client_id}] ASR result filtered (invalid): {text}"
                                        )
                                    else:
                                        await self.send_result(
                                            websocket,
                                            text,
                                            is_final=True,
                                            is_speeching=False,
                                            segment_id=segment_id,
                                        )
                                else:
                                    logger.warning(
                                        f"[Client {client_id}] Segment recognition failed"
                                    )
                            else:
                                logger.info(
                                    f"[Client {client_id}] Speech segment too short ({duration:.2f}s), skipping"
                                )

                            # Clear speech buffer for next segment
                            speech_buffer.clear()

                elif isinstance(message, str):
                    # Handle text messages (commands, etc.)
                    try:
                        data = json.loads(message)
                        await self.handle_command(
                            websocket, data, frame_buffer, speech_buffer, ns_processor, vad_processor
                        )
                    except json.JSONDecodeError:
                        await self.send_error(websocket, "Invalid JSON message", code=1)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[Client {client_id}] Connection closed")
        except Exception as e:
            logger.error(f"[Client {client_id}] Error: {e}\n{traceback.format_exc()}")
            await self.send_error(websocket, str(e), code=2)
        finally:
            logger.info(f"[Client {client_id}] Disconnected")

    async def handle_command(
        self,
        websocket,
        data: dict,
        frame_buffer: FrameBuffer,
        speech_buffer: SpeechBuffer,
        ns_processor: NSProcessor,
        vad_processor: VADProcessor,
    ):
        """Handle control commands from client."""
        command = data.get("command")

        if command == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

        elif command == "reset":
            frame_buffer.clear()
            speech_buffer.clear()
            vad_processor.reset()
            # Reinitialize NS processor to clear internal state
            if ns_processor.is_enabled():
                ns_processor.denoiser = RNNoise(sample_rate=ns_processor.sample_rate)
            await websocket.send(
                json.dumps({"type": "reset", "message": "State reset successfully"})
            )

        else:
            await self.send_error(websocket, f"Unknown command: {command}", code=3)

    async def send_result(
        self,
        websocket,
        text: str,
        is_final: bool,
        is_speeching: bool,
        segment_id: int = None,
    ):
        """Send recognition result to client."""
        result = {
            "type": "result",
            "text": text,
            "is_final": is_final,
            "is_speeching": is_speeching,
            "timestamp": int(time.time() * 1000),
        }
        if segment_id is not None:
            result["segment_id"] = segment_id

        await websocket.send(json.dumps(result, ensure_ascii=False))

    async def send_vad_event(
        self,
        websocket,
        speech_started: bool,
        duration: float = None,
        samples: int = None,
    ):
        """Send VAD event (speech start/end) to client."""
        event = {
            "type": "vad",
            "event": "speech_start" if speech_started else "speech_end",
            "timestamp": int(time.time() * 1000),
        }
        if duration is not None:
            event["duration"] = round(duration, 3)
        if samples is not None:
            event["samples"] = samples

        await websocket.send(json.dumps(event, ensure_ascii=False))

    async def send_error(self, websocket, message: str, code: int = 1):
        """Send error message to client."""
        error = {
            "type": "error",
            "message": message,
            "code": code,
            "timestamp": int(time.time() * 1000),
        }
        try:
            await websocket.send(json.dumps(error, ensure_ascii=False))
        except:
            pass

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.timeout,
        ):
            logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}/")
            await asyncio.Future()  # Run forever


# ============================================================================
# HTTP Service for Static Files
# ============================================================================


async def handle_http_request(request):
    """Handle HTTP requests for static file serving."""
    file_path = request.path.strip("/") or HTML_FILE

    # Security check - prevent directory traversal
    if ".." in file_path or file_path.startswith("/"):
        return web.Response(status=403, text="Forbidden")

    # Try to serve the requested file
    import os

    if os.path.exists(file_path):
        content_type = "text/html"
        if file_path.endswith(".js"):
            content_type = "application/javascript"
        elif file_path.endswith(".css"):
            content_type = "text/css"
        elif file_path.endswith(".json"):
            content_type = "application/json"

        with open(file_path, "rb") as f:
            content = f.read()

        return web.Response(
            body=content,
            content_type=content_type,
            headers={"Access-Control-Allow-Origin": "*"},
        )
    else:
        return web.Response(status=404, text="File not found")


async def start_http_server():
    """Start the HTTP server for static files."""
    app = web.Application()
    app.router.add_get("/{tail:.*}", handle_http_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HTTP_HOST, HTTP_PORT)
    await site.start()
    logger.info(f"HTTP server listening on http://{HTTP_HOST}:{HTTP_PORT}/")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Main entry point - start both WebSocket and HTTP servers."""
    print("=" * 60)
    print("ASR WebSocket Server Starting...")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("ASR WebSocket Server Starting...")
    logger.info("=" * 60)

    # Start HTTP server first (fast)
    print("Starting HTTP server...")
    logger.info("Starting HTTP server...")
    _ = asyncio.create_task(start_http_server())

    # Initialize WebSocket service (loads ASR model - this takes time)
    print("Initializing WebSocket service (this may take a while)...")
    logger.info("Initializing WebSocket service (this may take a while)...")
    ws_service = ASRWebSocketService()

    # Start WebSocket server (will run forever)
    print("Starting WebSocket server...")
    logger.info("Starting WebSocket server...")
    await ws_service.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown")
