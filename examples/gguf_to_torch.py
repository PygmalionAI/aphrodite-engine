import json
import os

import torch
import gguf
from sentencepiece import sentencepiece_model_pb2

def convert_to_state_dict(checkpoint, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = {}
    result = gguf.GGUFReader(checkpoint)
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]), encoding = 'utf-8')
    if architecture != "llama":
        print(f"Unsupported architecture {architecture}")
        return

    # write vocab
    vocab = sentencepiece_model_pb2.ModelProto()
    vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
    vocab.trainer_spec.model_type = 2 # BPE
    vocab.trainer_spec.vocab_size = vocab_size
    vocab.trainer_spec.byte_fallback = True
    vocab.normalizer_spec.remove_extra_whitespaces = False
    tokens = result.fields['tokenizer.ggml.tokens']
    scores = result.fields['tokenizer.ggml.scores']
    types = result.fields['tokenizer.ggml.token_type']
    for i in range(vocab_size):
        new_token = vocab.SentencePiece()
        new_token.piece = str(bytes(tokens.parts[tokens.data[i]]), encoding = 'utf-8')
        new_token.score = scores.parts[scores.data[i]]
        # llama.cpp tokentype is the same with sentencepiece token type
        new_token.type = int(types.parts[types.data[i]])
        vocab.pieces.append(new_token)
    with open(os.path.join(save_dir, "tokenizer.model"), 'wb') as f:
        f.write(vocab.SerializeToString())
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "legacy": False,
        "clean_up_tokenization_spaces": False,
    }
    if 'tokenizer.ggml.bos_token_id' in result.fields:
        tokenizer_config["bos_token"] = vocab.pieces[int(result.fields['tokenizer.ggml.bos_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.eos_token_id' in result.fields:
        tokenizer_config["eos_token"] = vocab.pieces[int(result.fields['tokenizer.ggml.eos_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.padding_token_id' in result.fields:
        tokenizer_config["pad_token"] = vocab.pieces[int(result.fields['tokenizer.ggml.padding_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.unknown_token_id' in result.fields:
        tokenizer_config["unk_token"] = vocab.pieces[int(result.fields['tokenizer.ggml.unknown_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.add_bos_token' in result.fields:
        tokenizer_config["add_bos_token"] = bool(result.fields['tokenizer.ggml.add_bos_token'].parts[-1])
    if 'tokenizer.ggml.add_eos_token' in result.fields:
        tokenizer_config["add_eos_token"] = bool(result.fields['tokenizer.ggml.add_eos_token'].parts[-1])
    if 'tokenizer.chat_template' in result.fields:
        tokenizer_config["chat_template"] = str(bytes(result.fields['tokenizer.chat_template'].parts[-1]), encoding="utf-8")
    json.dump(tokenizer_config, open(os.path.join(save_dir, "tokenizer_config.json"), 'w'), indent=2)

    # write config
    context_length = int(result.fields['llama.context_length'].parts[-1])
    n_layer = int(result.fields['llama.block_count'].parts[-1])
    n_head = int(result.fields['llama.attention.head_count'].parts[-1])
    n_local_heads = int(result.fields['llama.attention.head_count_kv'].parts[-1])
    intermediate_size = int(result.fields['llama.feed_forward_length'].parts[-1])
    norm_eps = float(result.fields['llama.attention.layer_norm_rms_epsilon'].parts[-1])
    dim = int(result.fields['llama.embedding_length'].parts[-1])
    kv_dim = dim // n_head * n_local_heads
    arch = "MixtralForCausalLM"
    if 'llama.expert_count' in result.fields:
        arch = "MixtralForCausalLM"
        name = "mixtral"
    else:
        arch = "LlamaForCausalLM"
        name = "llama"
    model_config= {
        "architectures": [arch],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": dim,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": context_length,
        "model_type": name,
        "num_attention_heads": n_head,
        "num_hidden_layers": n_layer,
        "num_key_value_heads": n_local_heads,
        "rms_norm_eps": norm_eps,
        "torch_dtype": "float16",
        "vocab_size": vocab_size
    }
    if 'llama.rope.freq_base' in result.fields:
        model_config['rope_theta'] = float(result.fields['llama.rope.freq_base'].parts[-1])
    if 'llama.expert_count' in result.fields:
        model_config['num_local_experts'] = int(result.fields['llama.expert_count'].parts[-1])
        model_config['num_experts_per_tok'] = int(result.fields['llama.expert_used_count'].parts[-1])
    json.dump(model_config, open(os.path.join(save_dir, "config.json"), 'w'), indent=2)

    # write tensor
    tensor_mapping = {
        "token_embd": ("model.embed_tokens", vocab_size),
        "output": ("lm_head", vocab_size),
        "output_norm": ("model.norm", -1),
        "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm", -1),
        "blk.{bid}.attn_q": ("model.layers.{bid}.self_attn.q_proj", dim),
        "blk.{bid}.attn_k": ("model.layers.{bid}.self_attn.k_proj", kv_dim),
        "blk.{bid}.attn_v": ("model.layers.{bid}.self_attn.v_proj", kv_dim),
        "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj", dim),
        "blk.{bid}.attn_rot_embd": ("model.layers.{bid}.self_attn.rotary_emb.inv_freq", -1),
        "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm", -1),
        "blk.{bid}.ffn_up": ("model.layers.{bid}.mlp.up_proj", intermediate_size),
        "blk.{bid}.ffn_down": ("model.layers.{bid}.mlp.down_proj", dim),
        "blk.{bid}.ffn_gate": ("model.layers.{bid}.mlp.gate_proj", intermediate_size),
        "blk.{bid}.ffn_up.{xid}": ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w3", intermediate_size),
        "blk.{bid}.ffn_down.{xid}": ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w2", dim),
        "blk.{bid}.ffn_gate.{xid}": ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w1", intermediate_size),
        "blk.{bid}.ffn_gate_inp": ("model.layers.{bid}.block_sparse_moe.gate", model_config.get('num_local_experts', 1)),
    }
    mapping = {}
    max_block_num = 200
    max_expert_num = 8
    for k, v in tensor_mapping.items():
        for i in range(max_block_num):
            for j in range(max_expert_num):
                fk = k.format(bid=i, xid=j)
                fv = v[0].format(bid=i, xid=j)
                if k not in mapping:
                    mapping[fk] = (fv, v[1])

    for ts in result.tensors:
        weight_type = torch.tensor(int(ts.tensor_type), dtype=torch.int)
        layer, suffix = ts.name.rsplit(".", 1)
        new_key, output_dim = mapping[layer]
        new_key += f".{suffix}"
        data = torch.tensor(ts.data)
        if output_dim != -1:
            data = data.view(output_dim, -1)
        if weight_type > 1:
            state_dict[new_key.replace("weight", "weight_type")] = weight_type
        state_dict[new_key] = data
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert GGUF checkpoints to torch')

    parser.add_argument('--input', type=str, help='The path to GGUF file')
    parser.add_argument('--output', type=str, help='The path to output directory')
    args = parser.parse_args()
    convert_to_state_dict(args.input, args.output)
