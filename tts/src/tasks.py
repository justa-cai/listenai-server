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
    from .model_worker import ModelWorkerPool

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
        server_config: "ServerConfig" = None,
        worker_pool: "ModelWorkerPool" = None
    ):
        """
        Initialize the TTS request task.

        Args:
            session: Session object
            model_config: Model configuration
            connection: Connection object for sending responses
            server_config: Server configuration (optional)
            worker_pool: Optional worker pool for parallel inference
        """
        self.session = session
        self.model_config = model_config
        self.connection = connection
        self.server_config = server_config
        self.audio_processor = AudioProcessor()
        self._cancelled = False
        self._model_cache: Optional[ModelCache] = None
        self._debug_audio_chunks = []  # Accumulate sent audio for debug
        self._worker_pool = worker_pool

    async def run(self) -> None:
        """Run the TTS task."""
        try:
            await self._update_state("processing")
            await self._send_progress("processing", 0.0, "Processing request")

            params = self.session.params

            # Execute TTS based on mode (use worker pool if available)
            if self._worker_pool:
                if params.get("mode") == "streaming":
                    await self._generate_streaming_with_worker(params)
                else:
                    await self._generate_non_streaming_with_worker(params)
            else:
                # Fallback to old single-threaded mode
                model = await self._get_model()
                if params.get("mode") == "streaming":
                    await self._generate_streaming(model, params)
                else:
                    await self._generate_non_streaming(model, params)

            await self._update_state("completed")

            # Save debug audio (all chunks sent to client for streaming mode)
            if self._debug_audio_chunks and not self._cancelled and self._is_debug_audio_enabled():
                combined_audio = b''.join(self._debug_audio_chunks)
                sample_rate = self.session.result.get('sample_rate', 16000)
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
        """Generate audio in streaming mode (fallback without worker pool) with real-time delivery."""
        await self._update_state("generating")
        await self._send_progress("generating", 0.1, "Generating audio...")

        sequence = 0
        total_samples = 0
        sample_rate = None

        # Create a queue for thread-safe chunk delivery
        chunk_queue = asyncio.Queue()

        def generate_audio_chunks():
            """Generator that yields audio chunks in a separate thread."""
            try:
                chunk_generator = model.generate_streaming(
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

                loop = asyncio.get_event_loop()

                for chunk in chunk_generator:
                    # Put chunk to queue from the executor thread
                    try:
                        asyncio.run_coroutine_threadsafe(
                            chunk_queue.put(('chunk', chunk)),
                            loop
                        ).result(timeout=30.0)
                    except Exception as e:
                        logger.error(f"Failed to put chunk to queue: {e}")
                        break

                # Signal completion
                try:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(('done', None)),
                        loop
                    ).result(timeout=5.0)
                except Exception as e:
                    logger.error(f"Failed to put done signal: {e}")

            except Exception as e:
                logger.error(f"Generation thread error: {e}")
                try:
                    loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(('error', str(e))),
                        loop
                    ).result(timeout=5.0)
                except:
                    pass

        try:
            # Run generation in executor
            loop = asyncio.get_event_loop()
            generation_future = loop.run_in_executor(None, generate_audio_chunks)

            # Process chunks as they arrive
            while True:
                if self._cancelled:
                    # Cancel the generation future if possible
                    generation_future.cancel()
                    break

                try:
                    # Wait for next chunk with timeout
                    msg_type, data = await asyncio.wait_for(chunk_queue.get(), timeout=60.0)

                    if msg_type == 'done':
                        break
                    elif msg_type == 'error':
                        raise RuntimeError(f"Generation failed: {data}")
                    elif msg_type == 'chunk':
                        chunk = data
                        # Get sample rate from model
                        if sample_rate is None:
                            sample_rate = model.tts_model.sample_rate

                        # Convert to PCM 16-bit
                        audio_bytes = self.audio_processor.to_pcm16(chunk)

                        # Send chunk immediately
                        await self._send_audio_chunk(
                            sequence,
                            audio_bytes,
                            sample_rate,
                            is_final=False
                        )

                        total_samples += len(chunk)
                        sequence += 1

                        # Update progress periodically
                        if sequence % 5 == 0:
                            await self._send_progress("generating", 0.1 + min(0.7, sequence * 0.01), f"Generated {sequence} chunks...")

                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for chunk, generation may be stuck")
                    break

            # Wait for generation to complete
            try:
                await generation_future
            except asyncio.CancelledError:
                pass

            # Store result
            self.session.result = {
                "duration": total_samples / (sample_rate or 16000),
                "sample_rate": sample_rate or 16000,
                "samples": total_samples,
                "chunks": sequence
            }

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def _generate_non_streaming(self, model, params: dict) -> None:
        """Generate audio in non-streaming mode (fallback without worker pool)."""
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

            # Check if cancelled after generation completes
            if self._cancelled:
                logger.info(f"Task {self.session.request_id} cancelled after generation")
                return

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

    async def _generate_streaming_with_worker(self, params: dict) -> None:
        """Generate audio in streaming mode using worker pool with real-time delivery."""
        await self._update_state("generating")
        await self._send_progress("generating", 0.1, "Generating audio...")

        # Get available worker
        worker = self._worker_pool.get_available_worker()
        if worker is None:
            worker = self._worker_pool.get_any_ready_worker()
        if worker is None:
            raise RuntimeError("No model workers available")

        logger.info(f"Using model worker {worker.id} for streaming request {self.session.request_id}")

        # Create queue for real-time chunk delivery
        chunk_queue = asyncio.Queue()
        sequence = 0
        total_samples = 0

        try:
            # Start generation task (runs in background)
            generation_task = asyncio.create_task(
                worker.generate_streaming_realtime(
                    chunk_queue=chunk_queue,
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
            )

            sample_rate = None

            # Process chunks as they arrive
            while True:
                # Check if cancelled
                if self._cancelled:
                    generation_task.cancel()
                    break

                try:
                    # Wait for next chunk with timeout
                    msg_type, chunk = await asyncio.wait_for(chunk_queue.get(), timeout=60.0)

                    if msg_type == 'done':
                        break
                    elif msg_type == 'chunk':
                        # Convert to PCM 16-bit
                        audio_bytes = self.audio_processor.to_pcm16(chunk)
                        self._debug_audio_chunks.append(audio_bytes)

                        # Send chunk immediately
                        await self._send_audio_chunk(
                            sequence,
                            audio_bytes,
                            sample_rate or 16000,  # Will be updated when we get first chunk
                            is_final=False
                        )
                        sequence += 1
                        total_samples += len(chunk)

                        # Update progress
                        if sequence % 5 == 0:
                            await self._send_progress("generating", 0.1 + min(0.7, sequence * 0.01), f"Generated {sequence} chunks...")

                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for chunk, generation may be stuck")
                    break

            # Wait for generation to complete
            try:
                sample_rate, _ = await generation_task
            except asyncio.CancelledError:
                pass

            # Store result
            self.session.result = {
                "duration": total_samples / (sample_rate or 16000),
                "sample_rate": sample_rate or 16000,
                "samples": total_samples,
                "chunks": sequence
            }

        except Exception as e:
            logger.error(f"Streaming generation with worker failed: {e}")
            raise

    async def _generate_non_streaming_with_worker(self, params: dict) -> None:
        """Generate audio in non-streaming mode using worker pool."""
        await self._update_state("generating")
        await self._send_progress("generating", 0.5, "Generating audio...")

        # Get available worker
        worker = self._worker_pool.get_available_worker()
        if worker is None:
            worker = self._worker_pool.get_any_ready_worker()
        if worker is None:
            raise RuntimeError("No model workers available")

        logger.info(f"Using model worker {worker.id} for non-streaming request {self.session.request_id}")

        try:
            wav, sample_rate, duration = await worker.generate(
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

            if self._cancelled:
                logger.info(f"Task {self.session.request_id} cancelled after generation")
                return

            await self._send_progress("encoding", 0.9, "Encoding audio...")

            # Convert to PCM 16-bit
            audio_bytes = self.audio_processor.to_pcm16(wav)

            # Save debug audio
            if self._is_debug_audio_enabled():
                _save_debug_audio(
                    audio_bytes,
                    sample_rate,
                    self.session.request_id,
                    "_non_streaming"
                )

            # Send complete audio
            await self._send_audio_full(
                audio_bytes,
                sample_rate,
                duration
            )

            # Store result
            self.session.result = {
                "duration": duration,
                "sample_rate": sample_rate,
                "samples": len(wav),
                "chunks": 1
            }

        except Exception as e:
            logger.error(f"Non-streaming generation with worker failed: {e}")
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
