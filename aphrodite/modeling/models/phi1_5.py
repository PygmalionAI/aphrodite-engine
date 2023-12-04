from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.attention import PagedAttention
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                              RowParallelLinear,
                                              QKVParallelLinear,
                                              LinearMethodBase)
from aphrodite.modeling.layers.rotary_embedding import get_rope
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from aphrodite.modeling.megatron.parallel_state import (
    get_tensor_model_parallel_world_size)
from aphrodite.modeling.hf_downloader import (default_weight_loader,
                                              hf_model_weights_iterator)
from aphrodite.common.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class PhiEmbedding(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(self, input_ids: torch.LongTensor):
        return self.wte(input_ids)


class PhiAttention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)

        # pylint: disable=C0103
        self.Wqkv = QKVParallelLinear(
            self.hidden_size,
            self.head_size,
            self.total_num_heads,
            linear_method=linear_method,
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            linear_method=linear_method,
        )

        scaling = self.head_size**-0.5
        rotary_dim = config.rotary_dim
        assert rotary_dim % 2 == 0

        # pylint: disable=C0301
        # Refer to:
        # https://huggingface.co/microsoft/phi-1_5/blob/d212a789620c380ff32ca1d1ee9943a777360987/modeling_phi.py#L518
        rope_theta = 10000
        max_position_embeddings = getattr(config, "n_positions", 2048)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = PagedAttention(self.num_heads, self.head_size, scaling)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata,
                                cache_event)
        output, _ = self.out_proj(attn_output)
        return output


class PhiMLP(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config.hidden_size

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            n_inner,
            linear_method=linear_method,
        )
        self.fc2 = RowParallelLinear(
            n_inner,
            config.hidden_size,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.activation_function, quant_config,
                              n_inner)

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class PhiLayer(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size,
                               eps=config.layer_norm_epsilon)
        self.mixer = PhiAttention(config, linear_method)
        self.mlp = PhiMLP(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)
        attn_outputs = self.mixer(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class PhiCausalLMHead(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size,
                               eps=config.layer_norm_epsilon)
        self.linear = ParallelLMHead(config.vocab_size,
                                     config.hidden_size,
                                     bias=True)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        hidden_states = self.ln(hidden_states)
        next_tokens = self.sampler(self.linear.weight, hidden_states,
                                   input_metadata, self.linear.bias)
        return next_tokens


class PhiModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.embd = PhiEmbedding(config)
        self.h = nn.ModuleList([
            PhiLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.embd(input_ids)
        for i in range(self.config.num_hidden_layers):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        return hidden_states


class PhiForCausalLM(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method

        self.transformer = PhiModel(config, linear_method)
        self.lm_head = PhiCausalLMHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        lm_logits = self.lm_head(hidden_states, input_metadata)
        return lm_logits

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            # pylint: disable=E1136
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
