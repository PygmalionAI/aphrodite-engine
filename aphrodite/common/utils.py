"""Utils."""
from os import path
import enum
from platform import uname
import uuid

import psutil
import torch
import asyncio
from functools import partial
from typing import (
    Awaitable,
    Callable,
    TypeVar,
)
from collections import OrderedDict
from typing import Any, Hashable, Optional

from aphrodite._C import cuda_utils

T = TypeVar("T")


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def __contains__(self, key: Hashable) -> bool:
        return key in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __getitem__(self, key: Hashable) -> Any:
        return self.get(key)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: Hashable) -> None:
        self.pop(key)

    def touch(self, key: Hashable) -> None:
        self.cache_to_end(key)

    def get(self, key: Hashable, default_value: Optional[Any] = None) -> int:
        if key in self.cache:
            value = self.cache[key]
            self.cache_to_end(key)
        else:
            value = default_value
        return value

    def put(self, key: Hashable, value: Any) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        self._remove_old_if_needed()

    def _on_remove(self, key: Hashable, value: Any):
        pass

    def remove_oldest(self):
        if not self.cache:
            return
        key, value = self.cache.popitem(last=False)
        self._on_remove(key, value)

    def _remove_old_if_needed(self) -> None:
        while len(self.cache) > self.capacity:
            self.remove_oldest()

    def pop(self, key: int, default_value: Optional[Any] = None) -> Any:
        run_on_remove = key in self.cache
        value = self.cache.pop(key, default_value)
        if run_on_remove:
            self._on_remove(key, value)
        return value

    def clear(self):
        while len(self.cache) > 0:
            self.remove_oldest()
        self.cache.clear()


def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97  # pylint: disable=invalid-name
    max_shared_mem = cuda_utils.get_device_attribute(
        cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu)
    return int(max_shared_mem)


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node or container in bytes."""

    memory_limit = psutil.virtual_memory().total

    for limit_file in [
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
            "/sys/fs/cgroup/memory.max"  # v2
    ]:
        if path.exists(limit_file):
            with open(limit_file) as f:
                content = f.read().strip()
                if content.isnumeric():  # v2 can have "max" as limit
                    memory_limit = min(memory_limit, int(content))

    return memory_limit


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()

def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Take a blocking function, and run it on an executor thread.
    
    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args, **kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)
    
    return _async_wrapper
