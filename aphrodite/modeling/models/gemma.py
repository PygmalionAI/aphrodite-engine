# coding=utf-8
# Copyright 2023 The PygmalionAI team.
# Copyright 2023 The vLLM team.
# Copyright (c) Google Inc.
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
"""Inference-only Gemma model compatible with HuggingFace weights."""
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import GemmaConfig

from aphrodite.attention import Attention, AttentionMetadata
from aphrodite.modeling.layers.activation import GeluAndMul
from aphrodite.modeling.layers.layernorm import RMSNorm
from aphrodite.modeling.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
from aphrodite.modeling.layers.rotary_embedding import get_rope
from aphrodite.modeling.layers.logits_processor import LogitsProcessor
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from aphrodite.distributed import (
    get_tensor_model_parallel_world_size, )
from aphrodite.modeling.sampling_metadata import SamplingMetadata
from aphrodite.modeling.hf_downloader import (
    default_weight_loader,
    hf_model_weights_iterator,
)
from aphrodite.common.sequence import SamplerOutput


@lru_cache(maxsize=None)
def _get_gemma_act_fn(
    hidden_act: Optional[str],
    hidden_activation: Optional[str],
) -> nn.Module:
    if hidden_activation is None:
        if hidden_act is not None:
            hidden_activation = hidden_act
        return GeluAndMul(approximate="none")
    elif hidden_activation == "gelu_pytorch_tanh":
        return GeluAndMul(approximate="tanh")
    elif hidden_activation == "gelu":
        return GeluAndMul(approximate="none")
    else:
        raise ValueError(f"Activation function {hidden_act} is not "
                         "supported for Gemma models.")


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Optional[str] = None,
        hidden_activation: Optional[str] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
        else:
            self.merge_weight = True
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                linear_method=linear_method,
            )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.act_fn = _get_gemma_act_fn(hidden_act, hidden_activation)

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


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                linear_method=linear_method,
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                linear_method=linear_method,
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                linear_method=linear_method,
            )
        else:
            self.merge_weight = True
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                linear_method=linear_method,
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        is_neox_style = (True if linear_method is None
                         or linear_method.quant_config.rope_style() is None
                         else linear_method.quant_config.rope_style())
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=is_neox_style,
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


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            linear_method=linear_method,
        )
        self.mlp = GemmaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=getattr(config, "hidden_activation", None),
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GemmaModel(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size,
                                                   linear_method=linear_method)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states *= self.normalizer

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


class GemmaForCausalLM(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = GemmaModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      linear_method=linear_method)
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.tokenizer_vocab_size)
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
            if "rotary_emb.inv_freq" in name:
                continue
            if "embed_tokens" in name:
                # Copy word embedding to lm_head
                head_name = name.replace("model.embed_tokens", "lm_head")
                if head_name in params_dict:
                    loaded_params.add(head_name)
                    lm_head_param = params_dict[head_name]
                    weight_loader = getattr(lm_head_param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(lm_head_param, loaded_weight)
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
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra layer for lora models.
                if "lm_head" in name and name not in params_dict:
                    continue
                # GemmaRMSNorm is different from Llama's in that it multiplies
                # (1 + weight) to the output, instead of just weight.
                if "norm.weight" in name:
                    loaded_weight += 1.0
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}")
