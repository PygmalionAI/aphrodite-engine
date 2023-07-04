"""
Multi-head Paged Attention by Woosuk et al. (vLLM) Copyright (c) 2023.
https://vllm.ai/
"""
from typing import Optional

import torch
import torch.nn as nn
from xformers import ops as xops

from aphrodite import attention_ops
from aphrodite import cache_ops
from aphrodite import pos_encoding_ops
from aphrodite.modeling.metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128]

class PagedAttention(nn.Module):
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The input 1D tensors
    can be split into three parts: the prompt tokens, the generation tokens, and the paddings.

    The prompts might have different lengths, while the generation tokens always have length 1.
    The paddings are appended to make the input length a multiple of 8, which is desirable for
    Tensor cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does not use the KV cache.
    2. Wait for the cache operations (e.g. swap, copy) to finish. The cache operations are issued
        by the cache engine before executing the forward pass of the model, and they are executed
        asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens. This operation reads
        the previous key and value tensors from the KV cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.attn_op = xops.fmha.cutlass.FwOp()

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                            f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: xops.AttentionBias,
    ) -> torch.Tensor:
        """
        TODO: The unsqueeze op may incur some CPU overhead. See if optimzation is possible.
        """
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
            op=self.attn_op,
        )
        # NOTE: Unnecessary copy. See if optimization is possible.
        output.copy_(out.squeeze(0))
        return output
    
    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:

        """
        NOTE: The query/value/key tensors must be sliced from a qkv tensor of shape
        [num_tokens, 3 * num_heads * head_size].
        """

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        output = torch.empty_like(query)

        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata.attn_bias,
            )

        if cache_event is not None:
            cache_event.wait()
        
        """
        Reshape the keys and values and store them in the cache.
        When key_cache and value_cache are not provided, the new key
        and value vector will not be cached.
        """

        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None and value_cache is not None):
            # The stride is 3 because the key/value are sliced from qkv.
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when generating tokens."
            )

            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata)
        
        # Reshapes the output tensor. The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)

class PagedAttentionWithRoPE(PagedAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        
        # NOTE: This assumes that we configure the default dtype when intializing the model. Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        pos_encoding_ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
