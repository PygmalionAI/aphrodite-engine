# coding=utf-8
# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
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
""" PyTorch LLaMA model."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from aphrodite.common.sequence import SequenceOutputs
from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.layers.activation import SiluAndMul
from aphrodite.modeling.layers.layernorm import RMSNorm
from aphrodite.modeling.layers.attention import PagedAttentionWithRoPE
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.hf_downloader import hf_model_weights_iterator, load_tensor_parallel_weights
from aphrodite.modeling.megatron.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from aphrodite.modeling.megatron.tensor_parallel import VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from aphrodite.common.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(hidden_size, 2 * intermediate_size,
                                                bias=False, gather_output=False,
                                                perform_initialization=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size,
                                            bias=False, input_is_parallel=True,
                                            perform_initialization=False)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is currently supported.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )