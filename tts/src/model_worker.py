"""Model worker for managing TTS inference in separate threads."""

import asyncio
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ModelWorker:
    """
    A model worker that runs in its own thread.

    Each worker has its own model instance and processes inference requests
    independently from other workers, enabling true parallel TTS generation.
    """

    def __init__(
        self,
        worker_id: int,
        model_config: ModelConfig,
        loop: asyncio.AbstractEventLoop
    ):
        """
        Initialize the model worker.

        Args:
            worker_id: Unique identifier for this worker
            model_config: Model configuration
            loop: Asyncio event loop for callbacks
        """
        self.worker_id = worker_id
        self.model_config = model_config
        self._loop = loop
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"model-worker-{worker_id}")
        self._ready = False
        self._busy = False

    async def start(self) -> None:
        """Start the worker and load the model."""
        logger.info(f"Starting model worker {self.worker_id}")
        await self._load_model()
        self._ready = True
        logger.info(f"Model worker {self.worker_id} ready")

    async def _load_model(self) -> None:
        """Load the model in this worker's thread."""
        def _load():
            from voxcpm import VoxCPM
            import os

            model_path = self.model_config.model_name
            if not os.path.isabs(model_path):
                if os.path.exists(model_path):
                    model_path = os.path.abspath(model_path)
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    potential_path = os.path.join(project_root, model_path)
                    if os.path.exists(potential_path):
                        model_path = potential_path

            # Check environment variables for denoiser configuration
            load_denoiser = os.getenv("TTS_LOAD_DENOISER", "true").lower() == "true"
            zipenhancer_path = os.getenv("TTS_ZIPENHANCER_PATH", None)

            logger.info(f"Worker {self.worker_id}: Loading model from {model_path}")
            logger.info(f"Worker {self.worker_id}: Denoiser enabled={load_denoiser}, path={zipenhancer_path or 'default'}")

            if load_denoiser and zipenhancer_path:
                model = VoxCPM.from_pretrained(
                    model_path,
                    load_denoiser=True,
                    zipenhancer_model_id=zipenhancer_path
                )
            elif load_denoiser:
                model = VoxCPM.from_pretrained(model_path, load_denoiser=True)
            else:
                model = VoxCPM.from_pretrained(model_path, load_denoiser=False)

            return model

        self._model = await self._loop.run_in_executor(self._executor, _load)

    async def generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0
    ) -> tuple:
        """
        Generate audio (non-streaming mode).

        Returns:
            Tuple of (wav_array, sample_rate, duration)
        """
        if not self._ready:
            raise RuntimeError(f"Worker {self.worker_id} not ready")

        self._busy = True
        try:
            def _generate():
                wav = self._model.generate(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise=denoise,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold
                )
                sample_rate = self._model.tts_model.sample_rate
                duration = len(wav) / sample_rate
                return wav, sample_rate, duration

            result = await self._loop.run_in_executor(self._executor, _generate)
            return result
        finally:
            self._busy = False

    async def generate_streaming_realtime(
        self,
        chunk_queue: asyncio.Queue,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0
    ) -> tuple:
        """
        Generate audio in true streaming mode with real-time chunk delivery.

        Args:
            chunk_queue: asyncio.Queue to put generated chunks
            ... (other params same as generate_streaming)

        Returns:
            Tuple of (sample_rate, total_samples)
        """
        if not self._ready:
            raise RuntimeError(f"Worker {self.worker_id} not ready")

        self._busy = True
        total_samples = 0
        sample_rate = None

        try:
            def _generate_streaming():
                """Generate streaming chunks and put to queue."""
                nonlocal sample_rate, total_samples

                chunk_generator = self._model.generate_streaming(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise=denoise,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold
                )

                sample_rate = self._model.tts_model.sample_rate

                for chunk in chunk_generator:
                    # Put chunk to queue (non-blocking with timeout)
                    try:
                        # Use run_coroutine_threadsafe to put to async queue from thread
                        asyncio.run_coroutine_threadsafe(
                            chunk_queue.put(('chunk', chunk)),
                            self._loop
                        ).result(timeout=30.0)
                        total_samples += len(chunk)
                    except Exception as e:
                        logger.error(f"Failed to put chunk to queue: {e}")
                        break

                # Signal completion
                try:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(('done', None)),
                        self._loop
                    ).result(timeout=5.0)
                except Exception as e:
                    logger.error(f"Failed to put done signal: {e}")

            # Start generation in executor
            future = self._loop.run_in_executor(self._executor, _generate_streaming)

            # Wait for completion and return metadata
            await future

            return sample_rate, total_samples

        finally:
            self._busy = False

    async def stop(self) -> None:
        """Stop the worker and release resources."""
        logger.info(f"Stopping model worker {self.worker_id}")
        self._ready = False
        # Use wait=False to shutdown immediately without waiting for pending tasks
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info(f"Model worker {self.worker_id} stopped")

    @property
    def is_ready(self) -> bool:
        """Check if the worker is ready."""
        return self._ready

    @property
    def is_busy(self) -> bool:
        """Check if the worker is currently processing."""
        return self._busy

    @property
    def id(self) -> int:
        """Get worker ID."""
        return self.worker_id


class ModelWorkerPool:
    """
    Pool of model workers for parallel TTS inference.
    """

    def __init__(self, num_workers: int, model_config: ModelConfig, loop: asyncio.AbstractEventLoop):
        """
        Initialize the worker pool.

        Args:
            num_workers: Number of worker threads
            model_config: Model configuration
            loop: Asyncio event loop
        """
        self.num_workers = num_workers
        self.model_config = model_config
        self._loop = loop
        self._workers: list[ModelWorker] = []
        self._ready = False

    async def start(self) -> None:
        """Start all workers in the pool."""
        logger.info(f"Starting model worker pool with {self.num_workers} workers")
        self._workers = [
            ModelWorker(i, self.model_config, self._loop)
            for i in range(self.num_workers)
        ]
        for worker in self._workers:
            await worker.start()
        self._ready = True
        logger.info("Model worker pool ready")

    async def stop(self) -> None:
        """Stop all workers."""
        for worker in self._workers:
            await worker.stop()
        self._workers.clear()
        self._ready = False

    def get_available_worker(self) -> Optional[ModelWorker]:
        """Get an available (not busy) worker."""
        for worker in self._workers:
            if worker.is_ready and not worker.is_busy:
                return worker
        return None

    def get_any_ready_worker(self) -> Optional[ModelWorker]:
        """Get any ready worker (for when all are busy, round-robin)."""
        ready_workers = [w for w in self._workers if w.is_ready]
        if not ready_workers:
            return None
        # Return the least busy one
        return min(ready_workers, key=lambda w: w.is_busy)

    @property
    def available_count(self) -> int:
        """Get number of available workers."""
        return sum(1 for w in self._workers if w.is_ready and not w.is_busy)

    @property
    def busy_count(self) -> int:
        """Get number of busy workers."""
        return sum(1 for w in self._workers if w.is_busy)

    @property
    def total_count(self) -> int:
        """Get total number of workers."""
        return len(self._workers)

    @property
    def is_ready(self) -> bool:
        """Check if pool is ready."""
        return self._ready

    def get_status(self) -> dict:
        """Get pool status."""
        return {
            "total": self.total_count,
            "available": self.available_count,
            "busy": self.busy_count,
            "ready": self.is_ready,
        }
