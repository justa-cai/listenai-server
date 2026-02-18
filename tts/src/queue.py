"""Task queue for managing concurrent TTS requests."""

import asyncio
from typing import Optional
from .tasks import TTSRequestTask


class TaskQueue:
    """
    Queue for managing TTS requests with concurrency control.

    Limits the number of concurrent TTS generation tasks and
    queues additional requests.
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize the task queue.

        Args:
            max_concurrent: Maximum number of concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue[TTSRequestTask] = asyncio.Queue()
        self._running_tasks: set[asyncio.Task] = set()
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
            num_workers: Number of worker coroutines to start
        """
        for _ in range(num_workers):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)

    async def stop(self) -> None:
        """Stop all worker coroutines."""
        for worker in self._workers:
            worker.cancel()
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _worker(self) -> None:
        """Worker coroutine that processes tasks from the queue."""
        while True:
            try:
                # Get task from queue
                task = await self._queue.get()

                # Create worker task
                async def run_task():
                    async with self._semaphore:
                        try:
                            await task.run()
                        except asyncio.CancelledError:
                            task.session.cancelled = True
                        except Exception:
                            # Error is handled within the task
                            pass
                        finally:
                            self._queue.task_done()

                worker_task = asyncio.create_task(run_task())
                self._running_tasks.add(worker_task)
                worker_task.add_done_callback(self._running_tasks.discard)

            except asyncio.CancelledError:
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
        # Cancel running tasks
        for task in list(self._running_tasks):
            if hasattr(task, 'request_id') and task.request_id == request_id:
                task.cancel()
                return True

        # Cancel queued tasks (need to check session)
        # Note: This is a simplified implementation
        # A full implementation would require tracking queued tasks by request_id
        return False

    def get_status(self) -> dict:
        """
        Get queue status.

        Returns:
            Dictionary with queue status
        """
        return {
            "pending": self.pending_count,
            "running": self.running_count,
            "available_slots": self.available_slots,
            "max_concurrent": self.max_concurrent,
        }
