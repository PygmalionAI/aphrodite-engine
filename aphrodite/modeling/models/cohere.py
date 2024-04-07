# coding=utf-8
# Copyright 2024 Cohere and the HuggingFace Inc. team. All rights reserved.
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

# This file is based on the LLama model definition file in transformers
"""PyTorch Cohere model."""
from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import CohereConfig

from aphrodite.attention import Attention, AttentionMetadata
from aphrodite.modeling.layers.activation import SiluAndMul
from aphrodite.modeling.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.rotary_embedding import get_rope
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from aphrodite.modeling.megatron.parallel_state import (
    get_tensor_model_parallel_world_size, )
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.modeling.hf_downloader import (
    default_weight_loader,
    hf_model_weights_iterator,
)
from aphrodite.common.sequence import SamplerOutput


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.variance_epsilon = eps

    def forward(self, hidden_states, residuals=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states -
                         mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        if self.bias is not None:
            hidden_states = hidden_states + self.bias.to(torch.float32)
        return hidden_states.to(input_dtype), residuals


# Copied from transformers.models.llama.modeling_llama.LlamaMLP Llama->Cohere
class CohereMLP(nn.Module):

    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
        else:
            self.merge_weight = True
            self.gate_up_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.intermediate_size] * 2,
                bias=False,
                linear_method=linear_method,
            )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if self.merge_weight:
            gate_up, _ = self.gate_up_proj(x)
        else:
            up, _ = self.up_proj(x)
            gate, _ = self.gate_proj(x)
            gate_up = torch.cat([gate, up], dim=-1)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class CohereAttention(nn.Module):

    def __init__(
        self,
        config: CohereConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                linear_method=linear_method,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                linear_method=linear_method,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                linear_method=linear_method,
            )
        else:
            self.merge_weight = True
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                linear_method=linear_method,
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=False,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.merge_weight:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
        else:
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class CohereDecoderLayer(nn.Module):

    def __init__(
        self,
        config: CohereConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CohereAttention(config, linear_method=linear_method)

        self.mlp = CohereMLP(config, linear_method=linear_method)
        self.input_layernorm = LayerNorm(config.hidden_size,
                                         eps=config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states_attention = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states_mlp = self.mlp(hidden_states)
        # Add everything together
        hidden_states = residual + hidden_states_attention + hidden_states_mlp

        return hidden_states, residual


class CohereModel(nn.Module):

    def __init__(
        self,
        config: CohereConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size,
                                                   linear_method=linear_method)
        self.layers = nn.ModuleList([
            CohereDecoderLayer(config, linear_method=linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CohereForCausalLM(nn.Module):

    def __init__(
        self,
        config: CohereConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      linear_method=linear_method)
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                scale=config.logit_scale)
        self.model = CohereModel(config, linear_method)
        self.sampler = Sampler()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if (self.linear_method is not None
                and not self.linear_method.quant_config.merge_weight()):
            stacked_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision,
                self.config):
            if "model.embed_tokens" in name:
                # Copy word embedding to lm_head
                head_name = name.replace("model.embed_tokens", "lm_head")
                if head_name in params_dict:
                    lm_head_param = params_dict[head_name]
                    weight_loader = getattr(lm_head_param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(lm_head_param, loaded_weight)
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "lm_head.weight" in name:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
