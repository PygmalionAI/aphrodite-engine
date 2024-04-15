# ruff: noqa

from __future__ import annotations

from typing import Sequence

from .constants import MODEL_ARCH, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES


class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",  # gptneox
            "transformer.wte",  # gpt2 gpt-j mpt refact qwen dbrx
            "transformer.word_embeddings",  # falcon
            "word_embeddings",  # bloom
            "model.embed_tokens",  # llama-hf
            "tok_embeddings",  # llama-pth
            "embeddings.word_embeddings",  # bert nomic-bert
            "language_model.embedding.word_embeddings",  # persimmon
            "wte",  # gpt2
            "transformer.embd.wte",  # phi2
            "model.tok_embeddings",  # internlm2
            "model.embedding",  # mamba-qbert
            "backbone.embedding",  # mamba
            "backbone.embeddings",  # mamba-hf
            "transformer.in_out_embed",  # Grok
        ),

        # Token type embeddings
        MODEL_TENSOR.TOKEN_TYPES: (
            "embeddings.token_type_embeddings",  # bert nomic-bert
        ),

        # Normalization of token embeddings
        MODEL_TENSOR.TOKEN_EMBD_NORM: (
            "word_embeddings_layernorm",  # bloom
            "embeddings.LayerNorm",  # bert
            "emb_ln",  # nomic-bert
        ),

        # Position embeddings
        MODEL_TENSOR.POS_EMBD: (
            "transformer.wpe",  # gpt2
            "embeddings.position_embeddings",  # bert
            "wpe",  # gpt2
        ),

        # Output
        MODEL_TENSOR.OUTPUT: (
            "embed_out",  # gptneox
            "lm_head",  # gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx
            "output",  # llama-pth bloom internlm2
            "word_embeddings_for_head",  # persimmon
            "lm_head.linear",  # phi2
        ),

        # Output norm
        MODEL_TENSOR.OUTPUT_NORM: (
            "gpt_neox.final_layer_norm",  # gptneox
            "transformer.ln_f",  # gpt2 gpt-j falcon
            "model.norm",  # llama-hf baichuan internlm2
            "norm",  # llama-pth
            "transformer.norm_f",  # mpt dbrx
            "ln_f",  # refact bloom qwen gpt2
            "language_model.encoder.final_layernorm",  # persimmon
            "model.final_layernorm",  # persimmon
            "lm_head.ln",  # phi2
            "model.norm_f",  # mamba-qbert
            "backbone.norm_f",  # mamba
            "transformer.rms_norm",  # Grok
        ),

        # Rope frequencies
        MODEL_TENSOR.ROPE_FREQS: (
            "rope.freqs",  # llama-pth
        ),
    }

    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",  # gptneox
            "transformer.h.{bid}.ln_1",  # gpt2 gpt-j refact qwen
            "transformer.blocks.{bid}.norm_1",  # mpt
            "transformer.h.{bid}.input_layernorm",  # falcon7b
            "h.{bid}.input_layernorm",  # bloom
            "transformer.h.{bid}.ln_mlp",  # falcon40b
            "model.layers.{bid}.input_layernorm",  # llama-hf
            "layers.{bid}.attention_norm",  # llama-pth
            "language_model.encoder.layers.{bid}.input_layernorm",  # persimmon
            "model.layers.{bid}.ln1",  # yi
            "h.{bid}.ln_1",  # gpt2
            "transformer.h.{bid}.ln",  # phi2
            "model.layers.layers.{bid}.norm",  # plamo
            "model.layers.{bid}.attention_norm",  # internlm2
            "model.layers.{bid}.norm",  # mamba-qbert
            "backbone.layers.{bid}.norm",  # mamba
            "transformer.decoder_layer.{bid}.rms_norm",  # Grok
            "transformer.blocks.{bid}.norm_attn_norm.norm_1",  # dbrx
        ),

        # Attention norm 2
        MODEL_TENSOR.ATTN_NORM_2: (
            "transformer.h.{bid}.ln_attn",  # falcon40b
        ),

        # Attention query-key-value
        MODEL_TENSOR.ATTN_QKV: (
            "gpt_neox.layers.{bid}.attention.query_key_value",  # gptneox
            "transformer.h.{bid}.attn.c_attn",  # gpt2 qwen
            "transformer.blocks.{bid}.attn.Wqkv",  # mpt
            "transformer.blocks.{bid}.norm_attn_norm.attn.Wqkv",  # dbrx
            "transformer.h.{bid}.self_attention.query_key_value",  # falcon
            "h.{bid}.self_attention.query_key_value",  # bloom
            "language_model.encoder.layers.{bid}.self_attention.query_key_value",  # persimmon
            "model.layers.{bid}.self_attn.query_key_value",  # persimmon
            "h.{bid}.attn.c_attn",  # gpt2
            "transformer.h.{bid}.mixer.Wqkv",  # phi2
            "encoder.layers.{bid}.attn.Wqkv",  # nomic-bert
        ),

        # Attention query
        MODEL_TENSOR.ATTN_Q: (
            "model.layers.{bid}.self_attn.q_proj",  # llama-hf
            "layers.{bid}.attention.wq",  # llama-pth
            "encoder.layer.{bid}.attention.self.query",  # bert
            "transformer.h.{bid}.attn.q_proj",  # gpt-j
            "model.layers.layers.{bid}.self_attn.q_proj",  # plamo
            "model.layers.{bid}.attention.wq",  # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.query"  # Grok
        ),

        # Attention key
        MODEL_TENSOR.ATTN_K: (
            "model.layers.{bid}.self_attn.k_proj",  # llama-hf
            "layers.{bid}.attention.wk",  # llama-pth
            "encoder.layer.{bid}.attention.self.key",  # bert
            "transformer.h.{bid}.attn.k_proj",  # gpt-j
            "model.layers.layers.{bid}.self_attn.k_proj",  # plamo
            "model.layers.{bid}.attention.wk",  # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.key"  # Grok
        ),

        # Attention value
        MODEL_TENSOR.ATTN_V: (
            "model.layers.{bid}.self_attn.v_proj",  # llama-hf
            "layers.{bid}.attention.wv",  # llama-pth
            "encoder.layer.{bid}.attention.self.value",  # bert
            "transformer.h.{bid}.attn.v_proj",  # gpt-j
            "model.layers.layers.{bid}.self_attn.v_proj",  # plamo
            "model.layers.{bid}.attention.wv",  # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.value"  # Grok
        ),

        # Attention output
        MODEL_TENSOR.ATTN_OUT: (
            "gpt_neox.layers.{bid}.attention.dense",  # gptneox
            "transformer.h.{bid}.attn.c_proj",  # gpt2 refact qwen
            "transformer.blocks.{bid}.attn.out_proj",  # mpt
            "transformer.h.{bid}.self_attention.dense",  # falcon
            "h.{bid}.self_attention.dense",  # bloom
            "model.layers.{bid}.self_attn.o_proj",  # llama-hf
            "layers.{bid}.attention.wo",  # llama-pth
            "encoder.layer.{bid}.attention.output.dense",  # bert
            "transformer.h.{bid}.attn.out_proj",  # gpt-j
            "language_model.encoder.layers.{bid}.self_attention.dense",  # persimmon
            "model.layers.{bid}.self_attn.dense",  # persimmon
            "h.{bid}.attn.c_proj",  # gpt2
            "transformer.h.{bid}.mixer.out_proj",  # phi2
            "model.layers.layers.{bid}.self_attn.o_proj",  # plamo
            "model.layers.{bid}.attention.wo",  # internlm2
            "encoder.layers.{bid}.attn.out_proj",  # nomic-bert
            "transformer.decoder_layer.{bid}.multi_head_attention.linear",  # Grok
            "transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",  # dbrx
        ),

        # Attention output norm
        MODEL_TENSOR.ATTN_OUT_NORM: (
            "encoder.layer.{bid}.attention.output.LayerNorm",  # bert
            "encoder.layers.{bid}.norm1",  # nomic-bert
            "transformer.decoder_layer.{bid}.rms_norm_1",  # Grok
            "transformer.blocks.{bid}.norm_attn_norm.norm_2",  # dbrx
        ),

        # Rotary embeddings
        MODEL_TENSOR.ATTN_ROT_EMBD: (
            "model.layers.{bid}.self_attn.rotary_emb.inv_freq",  # llama-hf
            "layers.{bid}.attention.inner_attention.rope.freqs",  # llama-pth
            "model.layers.layers.{bid}.self_attn.rotary_emb.inv_freq",  # plamo
            "transformer.h.{bid}.attn.rotary_emb.inv_freq",  # codeshell
        ),

        # Feed-forward norm
        MODEL_TENSOR.FFN_NORM: (
            "gpt_neox.layers.{bid}.post_attention_layernorm",  # gptneox
            "transformer.h.{bid}.ln_2",  # gpt2 refact qwen
            "h.{bid}.post_attention_layernorm",  # bloom
            "transformer.blocks.{bid}.norm_2",  # mpt
            "model.layers.{bid}.post_attention_layernorm",  # llama-hf
            "layers.{bid}.ffn_norm",  # llama-pth
            "language_model.encoder.layers.{bid}.post_attention_layernorm",  # persimmon
            "model.layers.{bid}.ln2",  # yi
            "h.{bid}.ln_2",  # gpt2
            "model.layers.{bid}.ffn_norm",  # internlm2
            "transformer.decoder_layer.{bid}.rms_norm_2",  # Grok
        ),
        MODEL_TENSOR.FFN_GATE_INP: (
            "layers.{bid}.feed_forward.gate",  # mixtral
            "model.layers.{bid}.block_sparse_moe.gate",  # mixtral
            "transformer.decoder_layer.{bid}.router",  # Grok
            "transformer.blocks.{bid}.ffn.router.layer",  # dbrx
        ),

        # Feed-forward up
        MODEL_TENSOR.FFN_UP: (
            "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",  # gptneox
            "transformer.h.{bid}.mlp.c_fc",  # gpt2
            "transformer.blocks.{bid}.ffn.up_proj",  # mpt
            "transformer.h.{bid}.mlp.dense_h_to_4h",  # falcon
            "h.{bid}.mlp.dense_h_to_4h",  # bloom
            "model.layers.{bid}.mlp.up_proj",  # llama-hf refact
            "layers.{bid}.feed_forward.w3",  # llama-pth
            "encoder.layer.{bid}.intermediate.dense",  # bert
            "transformer.h.{bid}.mlp.fc_in",  # gpt-j
            "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h",  # persimmon
            "model.layers.{bid}.mlp.dense_h_to_4h",  # persimmon
            "transformer.h.{bid}.mlp.w1",  # qwen
            "h.{bid}.mlp.c_fc",  # gpt2
            "transformer.h.{bid}.mlp.fc1",  # phi2
            "model.layers.{bid}.mlp.fc1",  # phi2
            "model.layers.layers.{bid}.mlp.up_proj",  # plamo
            "model.layers.{bid}.feed_forward.w3",  # internlm2
            "encoder.layers.{bid}.mlp.fc11",  # nomic-bert
            "model.layers.{bid}.mlp.c_fc",  # starcoder2
        ),
        MODEL_TENSOR.FFN_UP_EXP: (
            "layers.{bid}.feed_forward.experts.w3",  # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear_v",  # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.v1",  # dbrx
        ),

        # AWQ-activation gate
        MODEL_TENSOR.FFN_ACT: (
            "transformer.blocks.{bid}.ffn.act",  # mpt
        ),

        # Feed-forward gate
        MODEL_TENSOR.FFN_GATE: (
            "model.layers.{bid}.mlp.gate_proj",  # llama-hf refact
            "layers.{bid}.feed_forward.w1",  # llama-pth
            "transformer.h.{bid}.mlp.w2",  # qwen
            "model.layers.layers.{bid}.mlp.gate_proj",  # plamo
            "model.layers.{bid}.feed_forward.w1",  # internlm2
            "encoder.layers.{bid}.mlp.fc12",  # nomic-bert
        ),
        MODEL_TENSOR.FFN_GATE_EXP: (
            "layers.{bid}.feed_forward.experts.w1",  # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear",  # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.w1",  # dbrx
        ),

        # Feed-forward down
        MODEL_TENSOR.FFN_DOWN: (
            "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",  # gptneox
            "transformer.h.{bid}.mlp.c_proj",  # gpt2 refact qwen
            "transformer.blocks.{bid}.ffn.down_proj",  # mpt
            "transformer.h.{bid}.mlp.dense_4h_to_h",  # falcon
            "h.{bid}.mlp.dense_4h_to_h",  # bloom
            "model.layers.{bid}.mlp.down_proj",  # llama-hf
            "layers.{bid}.feed_forward.w2",  # llama-pth
            "encoder.layer.{bid}.output.dense",  # bert
            "transformer.h.{bid}.mlp.fc_out",  # gpt-j
            "language_model.encoder.layers.{bid}.mlp.dense_4h_to_h",  # persimmon
            "model.layers.{bid}.mlp.dense_4h_to_h",  # persimmon
            "h.{bid}.mlp.c_proj",  # gpt2
            "transformer.h.{bid}.mlp.fc2",  # phi2
            "model.layers.{bid}.mlp.fc2",  # phi2
            "model.layers.layers.{bid}.mlp.down_proj",  # plamo
            "model.layers.{bid}.feed_forward.w2",  # internlm2
            "encoder.layers.{bid}.mlp.fc2",  # nomic-bert
            "model.layers.{bid}.mlp.c_proj",  # starcoder2
        ),
        MODEL_TENSOR.FFN_DOWN_EXP: (
            "layers.{bid}.feed_forward.experts.w2",  # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear_1",  # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.w2",  # dbrx
        ),
        MODEL_TENSOR.ATTN_Q_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.q_layernorm",
            "model.layers.{bid}.self_attn.q_layernorm",  # persimmon
            "model.layers.{bid}.self_attn.q_norm",  # cohere
            "transformer.blocks.{bid}.attn.q_ln",  # sea-lion
        ),
        MODEL_TENSOR.ATTN_K_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.k_layernorm",
            "model.layers.{bid}.self_attn.k_layernorm",  # persimmon
            "model.layers.{bid}.self_attn.k_norm",  # cohere
            "transformer.blocks.{bid}.attn.k_ln",  # sea-lion
        ),
        MODEL_TENSOR.ROPE_FREQS: (
            "language_model.encoder.layers.{bid}.self_attention.rotary_emb.inv_freq",  # persimmon
        ),
        MODEL_TENSOR.LAYER_OUT_NORM: (
            "encoder.layer.{bid}.output.LayerNorm",  # bert
            "encoder.layers.{bid}.norm2",  # nomic-bert
            "transformer.decoder_layer.{bid}.rms_norm_3",  # Grok
        ),
        MODEL_TENSOR.SSM_IN: (
            "model.layers.{bid}.in_proj",
            "backbone.layers.{bid}.mixer.in_proj",
        ),
        MODEL_TENSOR.SSM_CONV1D: (
            "model.layers.{bid}.conv1d",
            "backbone.layers.{bid}.mixer.conv1d",
        ),
        MODEL_TENSOR.SSM_X: (
            "model.layers.{bid}.x_proj",
            "backbone.layers.{bid}.mixer.x_proj",
        ),
        MODEL_TENSOR.SSM_DT: (
            "model.layers.{bid}.dt_proj",
            "backbone.layers.{bid}.mixer.dt_proj",
        ),
        MODEL_TENSOR.SSM_A: (
            "model.layers.{bid}.A_log",
            "backbone.layers.{bid}.mixer.A_log",
        ),
        MODEL_TENSOR.SSM_D: (
            "model.layers.{bid}.D",
            "backbone.layers.{bid}.mixer.D",
        ),
        MODEL_TENSOR.SSM_OUT: (
            "model.layers.{bid}.out_proj",
            "backbone.layers.{bid}.mixer.out_proj",
        ),
    }

    mapping: dict[str, tuple[MODEL_TENSOR, str]]

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
        for bid in range(n_blocks):
            for tensor, keys in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue
                # TODO: make this configurable
                n_experts = 8
                for xid in range(n_experts):
                    tensor_name = TENSOR_NAMES[tensor].format(bid=bid, xid=xid)
                    self.mapping[tensor_name] = (tensor, tensor_name)
                    for key in keys:
                        key = key.format(bid=bid, xid=xid)
                        self.mapping[key] = (tensor, tensor_name)

    def get_type_and_name(
        self, key: str, try_suffixes: Sequence[str] = ()
    ) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[:-len(suffix)])
                if result is not None:
                    return result[0], result[1] + suffix
        return None

    def get_name(self, key: str,
                 try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes=try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type(
        self, key: str,
        try_suffixes: Sequence[str] = ()) -> MODEL_TENSOR | None:
        result = self.get_type_and_name(key, try_suffixes=try_suffixes)
        if result is None:
            return None
        return result[0]

    def __getitem__(self, key: str) -> str:
        try:
            return self.mapping[key][1]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return repr(self.mapping)


def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)
