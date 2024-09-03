import enum
import os
from functools import lru_cache
from typing import Optional, Type

import torch
from loguru import logger

from aphrodite.attention.backends.abstract import AttentionBackend
from aphrodite.common.utils import is_cpu, is_hip, is_openvino, is_tpu, is_xpu
from aphrodite.platforms import current_platform

APHRODITE_ATTENTION_BACKEND = "APHRODITE_ATTENTION_BACKEND"


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    OPENVINO = enum.auto()
    FLASHINFER = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()


@lru_cache(maxsize=None)
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_blocksparse: bool = False,
) -> Type[AttentionBackend]:

    if is_blocksparse:
        logger.info("Using BlocksparseFlashAttention backend.")
        from aphrodite.attention.backends.blocksparse_attn import (
            BlocksparseFlashAttentionBackend)
        return BlocksparseFlashAttentionBackend
    """Determine which attention backend to use and only import
    the selected backend module.
    """
    backend = which_attn_to_use(num_heads, head_size, num_kv_heads,
                                sliding_window, dtype, kv_cache_dtype,
                                block_size)
    if backend == _Backend.FLASH_ATTN:
        from aphrodite.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    if backend == _Backend.XFORMERS:
        logger.info("Using XFormers backend.")
        from aphrodite.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend
    elif backend == _Backend.ROCM_FLASH:
        logger.info("Using ROCmFlashAttention backend.")
        from aphrodite.attention.backends.rocm_flash_attn import (  # noqa: F401
            ROCmFlashAttentionBackend)
        return ROCmFlashAttentionBackend
    elif backend == _Backend.TORCH_SDPA:
        assert is_cpu(), RuntimeError(
            "Torch SDPA backend is only used for CPU devices.")
        logger.info("Using Torch SDPA backend.")
        from aphrodite.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
    elif backend == _Backend.OPENVINO:
        logger.info("Using OpenVINO attention backend.")
        from aphrodite.attention.backends.openvino import (
            OpenVINOAttentionBackend)
        return OpenVINOAttentionBackend
    elif backend == _Backend.IPEX:
        assert is_xpu(), RuntimeError(
            "IPEX attention backend is only used for the XPU device.")
        logger.info("Using IPEX attention backend.")
        from aphrodite.attention.backends.ipex_attn import IpexAttnBackend
        return IpexAttnBackend
    elif backend == _Backend.FLASHINFER:
        logger.info("Using Flashinfer backend.")
        from aphrodite.attention.backends.flashinfer import FlashInferBackend
        return FlashInferBackend
    elif backend == _Backend.PALLAS:
        logger.info("Using Pallas backend.")
        from aphrodite.attention.backends.pallas import PallasAttentionBackend
        return PallasAttentionBackend
    else:
        raise ValueError("Invalid attention backend.")


def which_attn_to_use(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
) -> _Backend:
    """Returns which flash attention backend to use."""

    # Default case.
    selected_backend = _Backend.FLASH_ATTN

    # Check the environment variable and override if specified
    backend_by_env_var: Optional[str] = os.getenv(APHRODITE_ATTENTION_BACKEND)
    if backend_by_env_var is not None:
        backend_members = _Backend.__members__
        if backend_by_env_var.upper() not in backend_members:
            raise ValueError(
                f"Invalid attention backend '{backend_by_env_var}'. "
                f"Available backends: {', '.join(backend_members)} ")
        selected_backend = _Backend[backend_by_env_var.upper()]
    if is_cpu():
        if selected_backend != _Backend.TORCH_SDPA:
            logger.info(f"Cannot use {selected_backend} backend on CPU.")
        return _Backend.TORCH_SDPA

    if is_openvino():
        if selected_backend != _Backend.OPENVINO:
            logger.info(f"Cannot use {selected_backend} backend on OpenVINO.")
        return _Backend.OPENVINO

    if is_xpu():
        if selected_backend != _Backend.IPEX:
            logger.info(f"Cannot use {selected_backend} backend on XPU.")
        return _Backend.IPEX

    if is_tpu():
        if selected_backend != _Backend.PALLAS:
            logger.info(f"Cannot use {selected_backend} backend on TPU.")
        return _Backend.PALLAS

    if is_hip():
        # AMD GPUs.
        selected_backend = (_Backend.ROCM_FLASH if selected_backend
                            == _Backend.FLASH_ATTN else selected_backend)
        if selected_backend == _Backend.ROCM_FLASH:
            if current_platform.get_device_capability()[0] != 9:
                # not Instinct series GPUs.
                logger.info("flash_attn is not supported on NAVI GPUs.")
        else:
            logger.info(f"{selected_backend} is not supported in AMD GPUs.")
        return _Backend.ROCM_FLASH

    # FlashAttn in NVIDIA GPUs.
    if selected_backend == _Backend.FLASH_ATTN:
        if current_platform.get_device_capability()[0] < 8:
            # Volta and Turing NVIDIA GPUs.
            logger.info(
                "Cannot use FlashAttention-2 backend for Volta and Turing "
                "GPUs.")
            selected_backend = _Backend.XFORMERS
        elif dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention-2 backend for dtype other than "
                "torch.float16 or torch.bfloat16.")
            selected_backend = _Backend.XFORMERS
        elif kv_cache_dtype is not None and kv_cache_dtype.startswith("fp8"):
            logger.info(
                "Cannot use FlashAttention-2 backend for FP8 KV cache.")
            selected_backend = _Backend.XFORMERS
        elif block_size % 16 != 0:
            logger.info(
                "Cannot use FlashAttention-2 backend for block size not "
                "divisible by 16.")
            selected_backend = _Backend.XFORMERS
        elif sliding_window is not None:
            logger.info(
                "Cannot use FlashAttention-2 backend due to sliding window.")
            selected_backend = _Backend.XFORMERS

    # FlashAttn is valid for the model, checking if the package is installed.
    if selected_backend == _Backend.FLASH_ATTN:
        try:
            import aphrodite_flash_attn  # noqa: F401

            from aphrodite.attention.backends.flash_attn import (  # noqa: F401
                FlashAttentionBackend)

            supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
            if head_size not in supported_sizes:
                logger.info(
                    "Cannot use FlashAttention-2 backend for head size "
                    f"{head_size}")
                selected_backend = _Backend.XFORMERS
        except ImportError:
            logger.info(
                "Cannot use FlashAttention-2 backend because the "
                "aphrodite_flash_attn package is not found. "
                "`pip install aphrodite-flash-attn` for better performance.")
            selected_backend = _Backend.XFORMERS

    return selected_backend
