# coding=utf-8
# Copyright 2023 Stability AI, EleutherAI, and The HuggingFace Inc. team.
# All rights reserved.
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
#
# This code is based off the following work:
# https://huggingface.co/stabilityai/stablelm-3b-4e1t/blob/main/modeling_stablelm_epoch.py
# https://huggingface.co/stabilityai/stablelm-3b-4e1t/blob/main/config.json
"""Inference-only StabeLM (https://github.com/Stability-AI/StableLM) model
compatible with HuggingFace weights."""

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.layers.activation import SiluAndMul
from aphrodite.modeling.layers.attention import PagedAttention
from aphrodite.modeling.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
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

KVCache = Tuple[torch.Tensor, torch.Tensor]


class StablelmMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
            self.up_proj = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
        else:
            self.merge_weight = True
            self.gate_up_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [config.intermediate_size] * 2,
                bias=False,
                linear_method=linear_method,
            )
        self.down_proj = RowParallelLinear(config.intermediate_size,
                                           config.hidden_size,
                                           bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_weight:
            gate_up, _ = self.gate_up_proj(x)
        else:
            up, _ = self.up_proj(x)
            gate, _ = self.gate_proj(x)
            gate_up = torch.cat([gate, up], dim=-1)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class StablelmAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_key_value_heads = config.num_key_value_heads
        if self.total_num_key_value_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_key_value_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_key_value_heads == 0
        self.num_key_value_heads = max(
            1, self.total_num_key_value_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings
        rope_pct = self.config.partial_rotary_factor
        self.rotary_ndims = int(self.head_dim * rope_pct)
        self.scaling = self.head_dim**-0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.qkv_bias = getattr(config, "use_qkv_bias", False)
        if (self.head_dim * self.num_heads * tp_size) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads (got "
                             f"`hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {self.num_heads}).")

        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.q_size,
                bias=self.qkv_bias,
                linear_method=linear_method,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.kv_size,
                bias=self.qkv_bias,
                linear_method=linear_method,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.kv_size,
                bias=self.qkv_bias,
                linear_method=linear_method,
            )
        else:
            self.merge_weight = True
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_key_value_heads,
                self.qkv_bias,
                linear_method=linear_method,
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_ndims = int(self.head_dim *
                                self.config.partial_rotary_factor)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_ndims,
            max_position=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_key_value_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
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
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class StablelmDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.self_attn = StablelmAttention(config)
        self.mlp = StablelmMLP(config, linear_method)
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


class StableLMEpochModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size,
                                                   linear_method=linear_method)
        self.layers = nn.ModuleList([
            StablelmDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # pylint: disable=unused-variable
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class StablelmForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = StableLMEpochModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      linear_method=linear_method)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head(hidden_states),
                                   sampling_metadata)
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
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision,
                self.config):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
