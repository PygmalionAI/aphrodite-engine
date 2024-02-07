# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The PygmalionAI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only T5 model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple
import math
import copy

import torch
from torch import nn
from transformers import T5Config

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.attention import PagedAttention
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.modeling.hf_downloader import (hf_model_weights_iterator,
                                              convert_pyslice_to_tensor)
from aphrodite.common.sequence import SamplerOutput

from aphrodite.common.logger import init_logger

KVCache = Tuple[torch.Tensor, torch.Tensor]

logger = init_logger(__name__)

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseAct(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    

class T5DenseGatedAct(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states
    

class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedAct(config)
        else:
            self.DenseReluDense = T5DenseAct(config)

        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)
        
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states
    

class T5Attention(nn.Module):
    def __init__(
            self,
            config: T5Config,
            has_relative_attention_bias: bool,
            is_cross: bool):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.is_cross = is_cross

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.has_relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)
        
        self.paged_attn = PagedAttention(
            self.n_heads, self.key_value_proj_dim, scale=1)
        
    
    @staticmethod
    def _relative_position_bucket(
        relative_position,
        bidirectional=True,
        num_buckets=32,
        max_distance=128):
        """Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are
        invalid. We use smaller buckets for small absolute relative_position and
        larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions
        <=-max_distance map to the same bucket. This should allow for more graceful
        generalization to longer sequences than the model has been trained on.
        
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
            
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position >
                                 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = - \
                torch.min(relative_position,
                          torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(
                relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small,
                                        relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device="cuda")[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device="cuda")[None, :]
        relative_position = memory_position - \
            context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_bias: Optional[torch.Tensor],
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        q = self.q(hidden_states)
        batch_size = hidden_states.shape[0]

        if not self.is_decoder:
            # Encoder self-attn

            def shape(states):
                """Projection."""
                return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            
            def unshape(states):
                """Reshape."""
                return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            
            q = shape(q)
            k = shape(self.k(hidden_states))
            v = shape(self.v(hidden_states))

            if position_bias is None:
                assert self.has_relative_attention_bias
                position_bias = self.compute_bias(
                    hidden_states.shape[1], hidden_states.shape[1])
            
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores + position_bias

            attn_weights = nn.functional.softmax(
                scores.float(),
                dim=-1).type_as(scores)
            attn_output = unshape(torch.matmul(attn_weights, v))
            attn_output = self.o(attn_output)

        elif not self.is_cross:
            # Decoder self-attn
            k = self.k(hidden_states)
            v = self.v(hidden_states)

            if position_bias is None:
                assert self.has_relative_attention_bias
                position_bias = self.compute_bias(
                    input_metadata.max_context_len, input_metadata.max_context_len)
            
            key_cache, value_cache = kv_cache

            attn_output = self.paged_attn(q, k, v, key_cache, value_cache,
                                          input_metadata, cache_event)
            attn_output = self.o(attn_output)
        
        else:
            # Decoder cross-attn
            assert position_bias is None
            assert self.has_relative_attention_bias == False

            k, v = kv_cache
            scores = torch.matmul(q, k.transpose(-1, -2))
            attn_weights = nn.functional.softmax(
                scores.float(),
                dim=-1).type_as(scores)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = self.o(attn_output)

        return (attn_output,) + (position_bias,)
    
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            is_cross=False)
        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_bias: Optional[torch.Tensor],
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            hidden_states=normed_hidden_states,
            position_bias=position_bias,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event)
        hidden_states = hidden_states + attention_output[0]
        return (hidden_states,) + (attention_output[1],)


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config,
            has_relative_attention_bias=False,
            is_cross=True)
        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states=normed_hidden_states,
            position_bias=position_bias,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event)
        hidden_states = hidden_states + attention_output[0]
        return (hidden_states,) + (attention_output[1],)
    

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias))
        
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
            self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        kv_cache: KVCache,
        cross_attention_kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ):
        
        self_attention_outputs = self.layer[0](
            hidden_states=hidden_states,
            position_bias=position_bias,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event)
        
        hidden_states = self_attention_outputs[0]
        self_attention_bias = self_attention_outputs[1]

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)
            
        if self.is_decoder:
            cross_attention_outputs = self.layer[1](
                hidden_states=hidden_states,
                kv_cache=cross_attention_kv_cache,
                position_bias=None,
                input_metadata=input_metadata,
                cache_event=cache_event)
            hidden_states = cross_attention_outputs[0]
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value)
        
        # Apply FF layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,) + (self_attention_bias,)

        return outputs
    

class T5Stack(nn.Module):
    def __init__(
            self,
            config: T5Config,
            embed_tokens: torch.Tensor):
        
        super().__init__()
        self.is_decoder = config.is_decoder
        self.embed_tokens = embed_tokens

        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=bool(i == 0))
            for i in range(config.num_layers)])
        
        self.final_layer_norm = T5LayerNorm(config.d_model,
                                            eps=config.layer_norm_epsilon)
        
    def forward(
            self,
            input_ids: torch.Tensor,
            kv_caches: List[KVCache],
            cross_attention_kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        
        hidden_states = self.embed_tokens(input_ids)
        position_bias = None

        for i, layer_module in enumerate(self.block):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]

            kv_cache = kv_caches[i] if self.is_decoder else None
            cross_attention_kv_cache = cross_attention_kv_caches[i] if self.is_decoder else None

            layer_outputs = layer_module(
                hidden_states,
                position_bias=position_bias,
                kv_cache=kv_cache,
                cross_attention_kv_cache=cross_attention_kv_cache,
                input_metadata=input_metadata,
                cache_event=cache_event)
            
            hidden_states = layer_outputs[0]

            # We share the position biases between the layers
            # The first layer to store them
            # layer_outputs = hidden_states, (self-attention position bias,)
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states
    

class T5ForConditionalGeneration(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.sampler = Sampler(config.vocab_size)


    # Only run decoder in the forward pass
    # We need to get cross_attention_kv_cache first by
    # calling model.prepare(...)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        cross_attention_kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
            cache_events=cache_events,
            cross_attention_kv_caches=cross_attention_kv_caches)
        
        sequence_output = decoder_outputs

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        next_tokens = self.sampler(
            self.shared.weight, sequence_output, positions)
        
        return next_tokens

    def prepare(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            input_metadata: InputMetadata) -> torch.Tensor:
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            kv_caches=None,
            input_metadata=input_metadata,
            cache_events=None,
            cross_attention_kv_caches=None)
        
        cross_attention_kv_caches: List[torch.Tensor] = []

        for block in self.decoder.block:
            cross_attention_layer = block.layer[1]
            k = cross_attention_layer.EncDecAttention.k(encoder_outputs)
            v = cross_attention_layer.EncDecAttention.v(encoder_outputs)
            cross_attention_kv_caches.append(torch.stack([k, v], dim=1))

            cross_attention_kv_caches_tensor = torch.stack(
                cross_attention_kv_caches, dim=0).transpose(0, 1)
            
        return cross_attention_kv_caches_tensor

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision):
            if 'EncDecAttention.relative_attention_bias' in name:
                continue
            assert name in state_dict

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            param = state_dict[name]
            assert param.shape == loaded_weight.shape, (
                f"{name} shape mismatch between model and checkpoint: "
                f"{param.shape} vs {loaded_weight.shape}")
            
            param.data.copy_(loaded_weight)
