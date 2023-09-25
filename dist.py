import functools
import os

import torch
from torch.distributed import broadcast, broadcast_object_list, is_initialized

def get_local_rank():
    return int(os.getenv('LOCAL_RANK', '0'))


def get_rank():
    return int(os.getenv('RANK', '0'))

def get_world_size():
    return int(os.getenv('WORLD_SIZE', '1'))

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_initialized():
            if get_rank() != 0:
                return None
        return func(*args, **kwargs)
    return wrapper

def master_only_and_broadcast_general(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_initialized():
            if get_rank() == 0:
                result = [func(*args, **kwargs)]
            else:
                result = [None]
            broadcast_object_list(result, src=0)
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper


def master_only_and_broadcast_tensor(func):
    @functools.wraps(func)
    def wrapper(*arg, size, dtype, **kwargs):
        if is_initialized():
            if get_rank() == 0:
                result = func(*args, **kwargs)
            else:
                result = torch.empty(size=size,
                                     dtype=dtype,
                                     device=get_local_rank())
            broadcast(result, src=0)
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper