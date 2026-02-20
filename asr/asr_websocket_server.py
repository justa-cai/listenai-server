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
VAD_SILENCE_FRAMES = (
    5  # Need 10 consecutive silence frames to exit speech state (hysteresis)
)

# ASR Configuration
ASR_MODEL_DIR = "FunAudioLLM/Fun-ASR-Nano-2512"
ASR_SAMPLE_RATE = 16000  # Hz
BUFFER_MAX_DURATION = 2.0  # seconds
ASR_LANGUAGE = "中文"  # Language for ASR: 中文, 英文, etc.
ASR_TEMP_DIR = "tmp"  # Directory for temporary wav files for debugging
ASR_KEEP_TEMP_FILES = True  # Keep temp wav files for debugging (set to False to delete)

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
    Voice Activity Detection using TenVad with hysteresis.

    Hysteresis mechanism:
    - Enter speech state immediately when any speech frame is detected
    - Exit speech state only after N consecutive silence frames
    """

    def __init__(
        self,
        hop_size: int = VAD_HOP_SIZE,
        threshold: float = VAD_THRESHOLD,
        silence_frames: int = VAD_SILENCE_FRAMES,
    ):
        self.hop_size = hop_size
        self.threshold = threshold
        self.silence_frames = (
            silence_frames  # Number of consecutive silence frames to exit speech
        )

        # Initialize TenVad
        try:
            self.ten_vad = TenVad(hop_size=hop_size, threshold=threshold)
            logger.info(
                f"TenVad initialized with hop_size={hop_size}, threshold={threshold}, silence_frames={silence_frames}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize TenVad: {e}")
            raise

        # VAD state machine
        self.is_speeching = False  # Current state
        self.consecutive_silence_count = 0  # Consecutive silence frame counter
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

        # Hysteresis state machine
        if is_speech_frame:
            # Any speech frame: enter speech state immediately
            if not self.is_speeching:
                logger.info(
                    f"Speech START at frame {self.frame_count}, prob: {probability:.3f}"
                )
            self.is_speeching = True
            self.consecutive_silence_count = 0
        else:
            # Silence frame
            if self.is_speeching:
                # In speech state: increment silence counter
                self.consecutive_silence_count += 1

                # Check if we should exit speech state
                if self.consecutive_silence_count >= self.silence_frames:
                    self.is_speeching = False
                    self.consecutive_silence_count = 0
                    logger.info(
                        f"Speech END at frame {self.frame_count} (after {self.silence_frames} consecutive silence frames)"
                    )
            else:
                # Already in silence state: do nothing
                pass

        # Debug logging for every frame (use DEBUG to reduce log spam)
        logger.debug(
            f"VAD Frame {self.frame_count}: prob={probability:.3f}, flag={speech_flag}, "
            f"is_speeching={self.is_speeching}, silence_count={self.consecutive_silence_count}/{self.silence_frames}"
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
        self.consecutive_silence_count = 0
        self.frame_count = 0


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
        vad_processor = VADProcessor()

        try:
            async for message in websocket:
                # Check if message is binary (audio data)
                if isinstance(message, bytes):
                    # Add to frame buffer and extract complete VAD frames
                    frames = frame_buffer.add(message)

                    # Debug logging (use DEBUG level to reduce spam)
                    logger.debug(
                        f"[Client {client_id}] Received {len(message)} bytes, extracted {len(frames)} frame(s), {frame_buffer.get_remaining_size()} bytes remaining"
                    )

                    # Process each complete VAD frame
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
                        elif was_speeching and is_speeching:
                            # Still in speech - continue adding
                            speech_buffer.add(chunk_float32)
                        elif was_speeching and not is_speeching:
                            # Speech segment ENDED - add final frame and process
                            speech_buffer.add(chunk_float32)
                            speech_buffer.stop()

                            duration = speech_buffer.get_duration()
                            samples = speech_buffer.total_samples

                            if duration >= 0.3:  # Minimum 0.3 seconds
                                logger.info(
                                    f"[Client {client_id}] Speech segment ended - "
                                    f"duration: {duration:.2f}s, samples: {samples}, "
                                    f"running segment recognition..."
                                )
                                result = self.asr_processor.process_segment_from_buffer(
                                    speech_buffer
                                )
                                if result:
                                    text, segment_id = result
                                    logger.info(
                                        f"[Client {client_id}] Segment recognition result: {text}"
                                    )
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
                            websocket, data, frame_buffer, speech_buffer, vad_processor
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
