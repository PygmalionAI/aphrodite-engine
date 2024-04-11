from functools import lru_cache
from typing import Type

import torch
from loguru import logger

from aphrodite.attention.backends.abstract import AttentionBackend
from aphrodite.common.utils import is_cpu, is_hip


@lru_cache(maxsize=None)
def get_attn_backend(dtype: torch.dtype) -> Type[AttentionBackend]:
    if _can_use_flash_attn(dtype):
        logger.info("Using FlashAttention backend.")
        from aphrodite.attention.backends.flash_attn import FlashAttentionBackend  # noqa: E501
        return FlashAttentionBackend
    elif is_cpu():
        logger.info("Using SDPA CPU backend.")
        from aphrodite.attention.backends.sdpa import TorchSDPABackend  # noqa: F401
        return TorchSDPABackend
    else:
        logger.info("Using XFormers backend.")
        from aphrodite.attention.backends.xformers import XFormersBackend  # noqa: F501
        return XFormersBackend


def _can_use_flash_attn(dtype: torch.dtype) -> bool:
    if is_hip():
        # AMD GPUs.
        logger.info("Cannot use FlashAttention backend for AMD GPUs.")
        return False
    if is_cpu():
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention backend for Volta and Turing "
                    "GPUs.")
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return False

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info(
            "Cannot use FlashAttention because the package is not found. "
            "Please install it for better performance.")
        return False
    return True
