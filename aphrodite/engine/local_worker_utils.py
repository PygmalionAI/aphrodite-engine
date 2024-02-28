import asyncio
import os
import traceback
import threading
import multiprocessing as mp
import uuid
from multiprocessing.connection import wait
from dataclasses import dataclass
from typing import Dict, List, TypeVar, Generic, Optional, Union

from aphrodite.common.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

_TERMINATE = "TERMINATE"


@dataclass
class Result(Generic[T]):
    """Result of the task dispatched to the worker."""

    task_id: uuid.UUID = None
    value: Optional[T] = None
    exception: Optional[BaseException] = None


class ResultFuture(threading.Event, Generic[T]):
    """Synchronous future for non-async case"""

    def __init__(self) -> None:
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]) -> None:
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value


def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    loop = future.get_loop()
    if result.exception is not None:
        loop.call_soon_threadsafe(future.set_exception, result.exception)
    else:
        loop.call_soon_threadsafe(future.set_result, result.value)


class ResultHandler(threading.Thread):
    """Handle results from all workers in the background thread"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = mp.Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    def run(self):
        for result in iter(self.result_queue.get, _TERMINATE):
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        for future in self.tasks.values():
            _set_future_result(
                future,
                Result(
                    exception=ChildProcessError("Worker Has Been Terminated")))

    def close(self):
        self.result_queue.put(_TERMINATE)


class WorkerMonitor(threading.Thread):
    """Monitor worker status in the background thread"""

    def __init__(self, workers: List["LocalWorkerAphrodite"],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers
        self.result_handler = result_handler
        self._close = False

    def run(self) -> None:
        dead_sentinels = wait([p.sentinel for p in self.workers])
        if self._close:
            return
        self._close = True

        # Kill / cleanup all workers
        for worker in self.workers:
            if worker.sentinel in dead_sentinels:
                worker.join(1)
            if worker.exitcode is not None and worker.exitcode != 0:
                logger.error(f"Worker {worker.name} pid {worker.pid} died, "
                             f"exit code: {worker.exitcode}")
        # Cleanup any remaining workers
        logger.info("Killing local Aphrodite worker processes")
        for worker in self.workers:
            worker.kill_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()

    def close(self):
        if self._close:
            return
        self._close = True
        logger.info("Terminating local Aphrodite worker processes")
        for worker in self.workers:
            worker.terminate_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()


class LocalWorkerAphrodite(mp.Process):
    """Local process wrapper for aphrodite.task_handler.worker
    for handling single-node multi-GPU tensor parallelism"""

    def __init__(self, result_handler: ResultHandler, *args, **kwargs) -> None:
        super().__init__(daemon=True)
        self._task_queue = mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.worker_args = args
        self.worker_kwargs = kwargs
        self.worker = None

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str, args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self._task_queue.put((task_id, method, args, kwargs))
        except Exception as e:
            del self.tasks[task_id]
            raise ChildProcessError("Worker Has Been Terminated") from e

    def execute_method(self, method: str, *args, **kwargs):
        future = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

    async def execute_method_async(self, method: str, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        try:
            self._task_queue.put(_TERMINATE)
        except ValueError:
            self.kill()
        self._task_queue.close()

    def kill_worker(self):
        self._task_queue.close()
        self.kill()

    def run(self) -> None:
        del self.tasks
        from aphrodite.task_handler.worker import Worker
        self.worker = Worker(*self.worker_args, **self.worker_kwargs)
        del self.worker_args
        del self.worker_kwargs

        logger.info(
            f"Worker {mp.current_process().name} pid {os.getpid()} ready; "
            "awaiting tasks")
        for items in iter(self._task_queue.get, _TERMINATE):
            output = None
            exception = None
            task_id, method, args, kwargs = items
            try:
                executor = getattr(self.worker, method)
                output = executor(*args, **kwargs)
            except BaseException as e:
                tb = traceback.format_exc()
                logger.error(
                    f"Exception in worker {mp.current_process().name} "
                    f"while processing method {method}: {e}, {tb}")
                exception = e
            self.result_queue.put(
                Result(task_id=task_id, value=output, exception=exception))

        logger.info(
            f"Worker {mp.current_process().name} pid {os.getpid()} terminated")
