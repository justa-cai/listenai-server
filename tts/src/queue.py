"""Task queue for managing concurrent TTS requests with worker pool."""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from .tasks import TTSRequestTask

if TYPE_CHECKING:
    from .model_worker import ModelWorkerPool

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Queue for managing TTS requests with concurrency control using worker pool.

    Distributes TTS requests to available model workers for parallel processing.
    """

    def __init__(self, max_concurrent: int = 10, worker_pool: Optional["ModelWorkerPool"] = None):
        """
        Initialize the task queue.

        Args:
            max_concurrent: Maximum number of concurrent tasks (should match worker pool size)
            worker_pool: Optional model worker pool for parallel inference
        """
        self.max_concurrent = max_concurrent
        self._worker_pool = worker_pool
        self._queue: asyncio.Queue[TTSRequestTask] = asyncio.Queue()
        self._running_tasks: set[asyncio.Task] = set()
        self._running_tts_tasks: dict[asyncio.Task, TTSRequestTask] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._workers: list[asyncio.Task] = []

    async def put(self, task: TTSRequestTask) -> None:
        """
        Add a task to the queue.

        Args:
            task: TTS request task to queue
        """
        await self._queue.put(task)

    def put_nowait(self, task: TTSRequestTask) -> None:
        """
        Add a task to the queue without waiting.

        Args:
            task: TTS request task to queue
        """
        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            raise RuntimeError("Task queue is full")

    async def start(self, num_workers: int = 1) -> None:
        """
        Start worker coroutines.

        Args:
            num_workers: Number of worker coroutines to start (dispatcher tasks)
        """
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        logger.info(f"Started {num_workers} dispatcher workers")

    async def stop(self) -> None:
        """Stop all worker coroutines and running tasks."""
        running_count = len(self._running_tasks)
        if running_count > 0:
            logger.info(f"Cancelling {running_count} running task(s)...")
            for task in list(self._running_tasks):
                task.cancel()

        for worker in self._workers:
            worker.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=0.8
            )
        except asyncio.TimeoutError:
            logger.warning("Workers did not finish within timeout, forcing stop")

        self._workers.clear()
        self._running_tasks.clear()
        self._running_tts_tasks.clear()

    async def _worker(self, worker_id: int) -> None:
        """Dispatcher coroutine that processes tasks from the queue."""
        logger.debug(f"Dispatcher worker {worker_id} started")
        while True:
            try:
                # Get task from queue
                task = await self._queue.get()

                # Assign worker pool to task if available
                if self._worker_pool:
                    task._worker_pool = self._worker_pool

                # Create worker task
                async def run_task():
                    async with self._semaphore:
                        try:
                            await task.run()
                        except asyncio.CancelledError:
                            task.session.cancelled = True
                        except Exception:
                            pass
                        finally:
                            self._queue.task_done()
                            self._running_tts_tasks.pop(asyncio.current_task(), None)

                worker_task = asyncio.create_task(run_task())
                self._running_tasks.add(worker_task)
                self._running_tts_tasks[worker_task] = task
                worker_task.add_done_callback(self._running_tasks.discard)
                worker_task.add_done_callback(lambda t: self._running_tts_tasks.pop(t, None))

            except asyncio.CancelledError:
                logger.debug(f"Dispatcher worker {worker_id} cancelled")
                break

    @property
    def pending_count(self) -> int:
        """Get the number of pending tasks in the queue."""
        return self._queue.qsize()

    @property
    def running_count(self) -> int:
        """Get the number of currently running tasks."""
        return len(self._running_tasks)

    @property
    def total_capacity(self) -> int:
        """Get the total capacity (running + available slots)."""
        return self.max_concurrent

    @property
    def available_slots(self) -> int:
        """Get the number of available slots for new tasks."""
        return self.max_concurrent - len(self._running_tasks)

    async def cancel_task(self, request_id: str) -> bool:
        """
        Cancel a task by request ID.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if task was found and cancelled
        """
        for task in list(self._running_tasks):
            tts_task = self._running_tts_tasks.get(task)
            if tts_task and tts_task.session.request_id == request_id:
                task.cancel()
                return True
        return False

    async def cancel_all_running(self) -> list[str]:
        """
        Cancel all currently running tasks.

        Returns:
            List of request_ids that were cancelled
        """
        cancelled_ids = []
        for worker_task, tts_task in list(self._running_tts_tasks.items()):
            tts_task.cancel()
            tts_task.session.cancelled = True
            tts_task.session.state = "cancelled"
            cancelled_ids.append(tts_task.session.request_id)
        return cancelled_ids

    def clear_pending(self) -> list[TTSRequestTask]:
        """
        Clear all pending tasks from the queue.

        Returns:
            List of cleared tasks
        """
        cleared = []
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                cleared.append(task)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        return cleared

    def get_status(self) -> dict:
        """
        Get queue status.

        Returns:
            Dictionary with queue status
        """
        status = {
            "pending": self.pending_count,
            "running": self.running_count,
            "available_slots": self.available_slots,
            "max_concurrent": self.max_concurrent,
        }
        if self._worker_pool:
            status["worker_pool"] = self._worker_pool.get_status()
        return status
