# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.layers.activation import SiluAndMul
from aphrodite.modeling.layers.attention import PagedAttention
from aphrodite.modeling.layers.layernorm import RMSNorm
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
from aphrodite.transformers_utils.configs.qwen import QWenConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class QWenMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        if (linear_method is not None
                and not linear_method.quant_config.merge_weight()):
            self.merge_weight = False
            self.w2 = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                linear_method=linear_method,
            )
            self.w1 = ColumnParallelLinear(
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
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if self.merge_weight:
            gate_up, _ = self.gate_up_proj(x)
        else:
            up, _ = self.w1(x)
            gate, _ = self.w2(x)
            gate_up = torch.cat([gate, up], dim=-1)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.scaling = self.head_dim**-0.5

        is_neox_style = (True if linear_method is None
                         or linear_method.quant_config.rope_style() is None
                         else linear_method.quant_config.rope_style())
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)

        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):

    def __init__(
        self,
        config: QWenConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            linear_method=linear_method,
        )

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            linear_method=linear_method,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class QWenModel(nn.Module):

    def __init__(
        self,
        config: QWenConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = VocabParallelEmbedding(config.vocab_size,
                                          config.hidden_size,
                                          linear_method=linear_method)
        self.h = nn.ModuleList([
            QWenBlock(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        residual = None
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


class QWenLMHeadModel(nn.Module):

    def __init__(
        self,
        config: QWenConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.transformer = QWenModel(config, linear_method)
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
        hidden_states = self.transformer(input_ids, positions, kv_caches,
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
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
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
