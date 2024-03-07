"""Attention layer."""
from typing import List, Optional

import torch
import torch.nn as nn

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.common.utils import is_hip
from aphrodite.common.logger import init_logger

logger = init_logger(__name__, deduplicate=True)


class Attention(nn.Module):
    """Attention layer.
    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:
    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        if (not is_hip() and torch.cuda.get_device_capability()[0] >= 8 and
                torch.get_default_dtype() in (torch.float16, torch.bfloat16)):
            # Ampere or later NVIDIA GPUs.
            # NOTE: FlashAttention does not support FP32.
            logger.info("Using FlashAttention backend.")
            from aphrodite.modeling.layers.attention.backends.flash_attn import FlashAttentionBackend
            self.backend = FlashAttentionBackend(num_heads, head_size, scale,
                                                 num_kv_heads, alibi_slopes,
                                                 sliding_window)
        else:
            # Turing, Volta, and Pascal NVIDIA GPUs or AMD GPUs.
            # Or FP32 on any GPU.
            logger.info("Using xFormers backend.")
            from aphrodite.modeling.layers.attention.backends.xformers import XFormersBackend
            self.backend = XFormersBackend(num_heads, head_size, scale,
                                           num_kv_heads, alibi_slopes,
                                           sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        return self.backend.forward(query, key, value, key_cache, value_cache,
                                    input_metadata)
