"""TTS request tasks."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from .session import Session
from .config import ModelConfig
from .audio_processor import AudioProcessor
from .connection import Connection
from .model_cache import ModelCache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ServerConfig

logger = logging.getLogger(__name__)

# Debug directory for saving generated audio
DEBUG_AUDIO_DIR = Path("./tmp/debug_audio")

def _save_debug_audio(audio_data: bytes, sample_rate: int, request_id: str, suffix: str = "") -> None:
    """Save audio data to debug directory as WAV file."""
    try:
        DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{request_id}{suffix}.wav"
        filepath = DEBUG_AUDIO_DIR / filename

        # Create WAV file
        num_samples = len(audio_data) // 2  # PCM 16-bit
        wav_header = bytearray()

        # RIFF header
        wav_header.extend(b"RIFF")
        wav_header.extend((36 + num_samples * 2).to_bytes(4, 'little'))
        wav_header.extend(b"WAVE")

        # fmt chunk
        wav_header.extend(b"fmt ")
        wav_header.extend((16).to_bytes(4, 'little'))  # Subchunk1Size
        wav_header.extend((1).to_bytes(2, 'little'))   # AudioFormat (PCM)
        wav_header.extend((1).to_bytes(2, 'little'))   # NumChannels
        wav_header.extend(sample_rate.to_bytes(4, 'little'))  # SampleRate
        wav_header.extend((sample_rate * 2).to_bytes(4, 'little'))  # ByteRate
        wav_header.extend((2).to_bytes(2, 'little'))   # BlockAlign
        wav_header.extend((16).to_bytes(2, 'little'))  # BitsPerSample

        # data chunk
        wav_header.extend(b"data")
        wav_header.extend((num_samples * 2).to_bytes(4, 'little'))  # Subchunk2Size
        wav_header.extend(audio_data)

        with open(filepath, "wb") as f:
            f.write(wav_header)

        # Also save metadata
        meta_filepath = DEBUG_AUDIO_DIR / f"{filename}.txt"
        with open(meta_filepath, "w", encoding="utf-8") as f:
            f.write(f"Request ID: {request_id}\n")
            f.write(f"Sample Rate: {sample_rate}\n")
            f.write(f"Channels: 1\n")
            f.write(f"Bits Per Sample: 16\n")
            f.write(f"Data Length: {len(audio_data)} bytes\n")
            f.write(f"Samples: {num_samples}\n")
            f.write(f"Duration: {num_samples / sample_rate:.2f}s\n")

        logger.info(f"Saved debug audio to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save debug audio: {e}")

# Debug directory for saving generated audio
DEBUG_AUDIO_DIR = Path("./tmp/debug_audio")


class TTSRequestTask:
    """
    Task for processing a TTS request.

    Handles both streaming and non-streaming audio generation.
    """

    def __init__(
        self,
        session: Session,
        model_config: ModelConfig,
        connection: Connection,
        server_config: "ServerConfig" = None
    ):
        """
        Initialize the TTS request task.

        Args:
            session: Session object
            model_config: Model configuration
            connection: Connection object for sending responses
            server_config: Server configuration (optional)
        """
        self.session = session
        self.model_config = model_config
        self.connection = connection
        self.server_config = server_config
        self.audio_processor = AudioProcessor()
        self._cancelled = False
        self._model_cache: Optional[ModelCache] = None
        self._debug_audio_chunks = []  # Accumulate sent audio for debug

    async def run(self) -> None:
        """Run the TTS task."""
        try:
            await self._update_state("processing")
            await self._send_progress("processing", 0.0, "Processing request")

            # Get model
            model = await self._get_model()

            params = self.session.params

            # Execute TTS based on mode
            if params.get("mode") == "streaming":
                await self._generate_streaming(model, params)
            else:
                await self._generate_non_streaming(model, params)

            await self._update_state("completed")

            # Save debug audio (all chunks sent to client for streaming mode)
            if self._debug_audio_chunks and not self._cancelled and self._is_debug_audio_enabled():
                combined_audio = b''.join(self._debug_audio_chunks)
                sample_rate = self.session.result.get('sample_rate', 24000)
                _save_debug_audio(
                    combined_audio,
                    sample_rate,
                    self.session.request_id,
                    "_sent_streaming"
                )

            await self._send_complete()

        except asyncio.CancelledError:
            self._cancelled = True
            await self._update_state("cancelled")
            await self._send_complete(cancelled=True)
        except Exception as e:
            logger.exception(f"TTS generation failed: {e}")
            await self._update_state("failed")
            await self._send_error(str(e))

    async def _get_model(self):
        """Get the VoxCPM model instance."""
        if self._model_cache is None:
            self._model_cache = await ModelCache.get_instance()
        return await self._model_cache.get_model(self.model_config)

    async def _generate_streaming(self, model, params: dict) -> None:
        """Generate audio in streaming mode."""
        await self._update_state("generating")
        await self._send_progress("generating", 0.1, "Generating audio...")

        sequence = 0
        total_samples = 0

        # Create a generator function that runs in executor
        def generate_audio_chunks():
            """Generator that yields audio chunks."""
            return model.generate_streaming(
                text=params["text"],
                prompt_wav_path=params.get("prompt_wav_path"),
                prompt_text=params.get("prompt_text"),
                cfg_value=params.get("cfg_value", self.model_config.default_cfg_value),
                inference_timesteps=params.get(
                    "inference_timesteps",
                    self.model_config.default_inference_timesteps
                ),
                normalize=params.get("normalize", self.model_config.default_normalize),
                denoise=params.get("denoise", self.model_config.default_denoise),
                retry_badcase=params.get(
                    "retry_badcase",
                    self.model_config.default_retry_badcase
                ),
                retry_badcase_max_times=params.get(
                    "retry_badcase_max_times",
                    self.model_config.default_retry_badcase_max_times
                ),
                retry_badcase_ratio_threshold=params.get(
                    "retry_badcase_ratio_threshold",
                    self.model_config.default_retry_badcase_ratio_threshold
                )
            )

        try:
            # Run the generator in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            chunk_iterator = await loop.run_in_executor(None, generate_audio_chunks)

            # Accumulate all audio bytes for debug saving
            all_audio_bytes = bytearray()

            for chunk in chunk_iterator:
                if self._cancelled:
                    break

                # Convert to PCM 16-bit
                audio_bytes = self.audio_processor.to_pcm16(chunk)
                all_audio_bytes.extend(audio_bytes)

                # Send audio chunk
                await self._send_audio_chunk(
                    sequence,
                    audio_bytes,
                    model.tts_model.sample_rate,
                    is_final=False
                )

                total_samples += len(chunk)
                sequence += 1

                # Yield control to event loop periodically
                await asyncio.sleep(0)

            # Save debug audio (complete audio)
            if not self._cancelled and all_audio_bytes and self._is_debug_audio_enabled():
                _save_debug_audio(
                    bytes(all_audio_bytes),
                    model.tts_model.sample_rate,
                    self.session.request_id,
                    "_streaming"
                )

            # Store result
            self.session.result = {
                "duration": total_samples / model.tts_model.sample_rate,
                "sample_rate": model.tts_model.sample_rate,
                "samples": total_samples,
                "chunks": sequence
            }

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def _generate_non_streaming(self, model, params: dict) -> None:
        """Generate audio in non-streaming mode."""
        await self._update_state("generating")
        await self._send_progress("generating", 0.5, "Generating audio...")

        try:
            wav = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate(
                    text=params["text"],
                    prompt_wav_path=params.get("prompt_wav_path"),
                    prompt_text=params.get("prompt_text"),
                    cfg_value=params.get(
                        "cfg_value",
                        self.model_config.default_cfg_value
                    ),
                    inference_timesteps=params.get(
                        "inference_timesteps",
                        self.model_config.default_inference_timesteps
                    ),
                    normalize=params.get(
                        "normalize",
                        self.model_config.default_normalize
                    ),
                    denoise=params.get(
                        "denoise",
                        self.model_config.default_denoise
                    ),
                    retry_badcase=params.get(
                        "retry_badcase",
                        self.model_config.default_retry_badcase
                    ),
                    retry_badcase_max_times=params.get(
                        "retry_badcase_max_times",
                        self.model_config.default_retry_badcase_max_times
                    ),
                    retry_badcase_ratio_threshold=params.get(
                        "retry_badcase_ratio_threshold",
                        self.model_config.default_retry_badcase_ratio_threshold
                    )
                )
            )

            await self._send_progress("encoding", 0.9, "Encoding audio...")

            # Convert to PCM 16-bit
            audio_bytes = self.audio_processor.to_pcm16(wav)

            # Save debug audio
            if self._is_debug_audio_enabled():
                _save_debug_audio(
                    audio_bytes,
                    model.tts_model.sample_rate,
                    self.session.request_id,
                    "_non_streaming"
                )

            # Send complete audio
            await self._send_audio_full(
                audio_bytes,
                model.tts_model.sample_rate,
                len(wav) / model.tts_model.sample_rate
            )

            # Store result
            self.session.result = {
                "duration": len(wav) / model.tts_model.sample_rate,
                "sample_rate": model.tts_model.sample_rate,
                "samples": len(wav),
                "chunks": 1
            }

        except Exception as e:
            logger.error(f"Non-streaming generation failed: {e}")
            raise

    async def _update_state(self, state: str) -> None:
        """Update the session state."""
        self.session.state = state

    async def _send_progress(
        self,
        state: str,
        progress: float,
        message: str
    ) -> None:
        """Send a progress update."""
        try:
            await self.connection.send_progress(
                self.session.request_id,
                state,
                progress,
                message
            )
        except Exception as e:
            logger.warning(f"Failed to send progress: {e}")

    async def _send_audio_chunk(
        self,
        sequence: int,
        audio_bytes: bytes,
        sample_rate: int,
        is_final: bool
    ) -> None:
        """Send an audio chunk in streaming mode."""
        try:
            # Accumulate for debug saving
            self._debug_audio_chunks.append(audio_bytes)

            await self.connection.send_binary_frame(
                msg_type=Connection.FRAME_TYPE_STREAMING_CHUNK,
                metadata={
                    "request_id": self.session.request_id,
                    "sequence": sequence,
                    "sample_rate": sample_rate,
                    "is_final": is_final
                },
                audio_data=audio_bytes
            )
        except Exception as e:
            logger.warning(f"Failed to send audio chunk: {e}")

    async def _send_audio_full(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        duration: float
    ) -> None:
        """Send complete audio in non-streaming mode."""
        try:
            # Save debug audio (sent to client)
            if self._is_debug_audio_enabled():
                _save_debug_audio(
                    audio_bytes,
                    sample_rate,
                    self.session.request_id,
                    "_sent"
                )

            await self.connection.send_binary_frame(
                msg_type=Connection.FRAME_TYPE_NON_STREAMING,
                metadata={
                    "request_id": self.session.request_id,
                    "sample_rate": sample_rate,
                    "duration": duration
                },
                audio_data=audio_bytes
            )
        except Exception as e:
            logger.warning(f"Failed to send audio: {e}")

    async def _send_complete(self, cancelled: bool = False) -> None:
        """Send a completion message."""
        try:
            result = self.session.result or {"cancelled": cancelled}
            await self.connection.send_complete(
                self.session.request_id,
                result
            )
        except Exception as e:
            logger.warning(f"Failed to send complete: {e}")

    async def _send_error(self, message: str) -> None:
        """Send an error message."""
        try:
            await self.connection.send_error(
                "GENERATION_FAILED",
                message
            )
        except Exception as e:
            logger.warning(f"Failed to send error: {e}")

    def _is_debug_audio_enabled(self) -> bool:
        """Check if debug audio saving is enabled."""
        return self.server_config is not None and getattr(self.server_config, 'debug_audio', False)

    def cancel(self) -> None:
        """Cancel the task."""
        self._cancelled = True
