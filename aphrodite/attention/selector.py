import enum
import os
from functools import lru_cache
from typing import Optional, Type

import torch
from loguru import logger
from aphrodite.attention.backends.abstract import AttentionBackend
from aphrodite.common.utils import is_cpu, is_hip

APHRODITE_ATTENTION_BACKEND = "APHRODITE_ATTENTION_BACKEND"


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()


@lru_cache(maxsize=None)
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
) -> Type[AttentionBackend]:
    backend = _which_attn_to_use(num_heads, head_size, num_kv_heads,
                                 sliding_window, dtype, kv_cache_dtype,
                                 block_size)
    if backend == _Backend.FLASH_ATTN:
        logger.info("Using FlashAttention backend.")
        from aphrodite.attention.backends.flash_attn import \
            FlashAttentionBackend  # noqa: F401
        return FlashAttentionBackend
    elif backend == _Backend.XFORMERS:
        logger.info("Using XFormers backend.")
        from aphrodite.attention.backends.xformers import \
            XFormersBackend  # noqa: F401
        return XFormersBackend
    elif backend == _Backend.ROCM_FLASH:
        logger.info("Using ROCmFlashAttention backend.")
        from aphrodite.attention.backends.rocm_flash_attn import \
            ROCmFlashAttentionBackend  # noqa: F401
        return ROCmFlashAttentionBackend
    elif backend == _Backend.TORCH_SDPA:
        logger.info("Using Torch SDPA backend.")
        from aphrodite.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
    elif backend == _Backend.FLASHINFER:
        logger.info("Using Flashinfer backend.")
        logger.warning("Eager mode is enforced for the Flashinfer backend. ")
        from aphrodite.attention.backends.flashinfer import FlashInferBackend
        return FlashInferBackend
    else:
        raise ValueError("Invalid attention backend.")


def _which_attn_to_use(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
) -> _Backend:
    """Returns which flash attention backend to use."""
    if is_cpu():
        return _Backend.TORCH_SDPA

    if is_hip():
        # AMD GPUs.
        if torch.cuda.get_device_capability()[0] != 9:
            # not Instinct series GPUs.
            logger.info("flash_atten is not supported on NAVI GPUs.")
        return _Backend.ROCM_FLASH

    # NVIDIA GPUs.
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention backend for Volta and Turing "
                    "GPUs.")
        return _Backend.XFORMERS

    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return _Backend.XFORMERS

    try:
        import vllm_flash_attn  # noqa: F401
    except ImportError:
        logger.info(
            "Cannot use FlashAttention-2 backend because the vllm_flash_attn "
            "package is not found. `pip install vllm-flash-attn` for better "
            "performance.")
        return _Backend.XFORMERS

    backend_by_env_var = os.getenv(APHRODITE_ATTENTION_BACKEND)
    if backend_by_env_var is not None:
        return _Backend[backend_by_env_var]

    # Default case.
    return _Backend.FLASH_ATTN
