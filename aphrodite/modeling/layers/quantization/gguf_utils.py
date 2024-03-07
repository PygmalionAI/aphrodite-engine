from __future__ import annotations

import json
import os
import shutil
import tempfile
from enum import IntEnum
from collections import OrderedDict
from typing import Any, Literal, NamedTuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
import torch
from transformers import LlamaTokenizer, GPT2Tokenizer
from transformers.convert_slow_tokenizer import import_protobuf
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
READER_SUPPORTED_VERSIONS = [2, GGUF_VERSION]


class GGMLQuantizationType(IntEnum):
    F32  = 0
    F16  = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23


QK_K = 256
# Items here are (block size, type size)
GGML_QUANT_SIZES = {
    GGMLQuantizationType.F32:  (1, 4),
    GGMLQuantizationType.F16:  (1, 2),
    GGMLQuantizationType.Q4_0: (32, 2 + 16),
    GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0: (32, 2 + 32),
    GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K: (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K: (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K: (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K: (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS: (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + 3 * QK_K // 8),
    GGMLQuantizationType.IQ1_S: (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL: (32, 2 + 32 // 2),
    GGMLQuantizationType.IQ3_S: (256, 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 + QK_K // 64),
    GGMLQuantizationType.IQ2_S: (256, 2 + QK_K // 4 + QK_K // 32 + QK_K // 32),
    GGMLQuantizationType.IQ4_XS: (256, 2 + 2 + QK_K // 64 + QK_K // 2),
}

class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32


class ReaderField(NamedTuple):
    # Offset to start of this field.
    offset: int

    # Name of the field (not necessarily from file data).
    name: str

    # Data parts. Some types have multiple components, such as strings
    # that consist of a length followed by the string data.
    parts: list[npt.NDArray[Any]] = []

    # Indexes into parts that we can call the actual data. For example
    # an array of strings will be populated with indexes to the actual
    # string data.
    data: list[int] = [-1]

    types: list[GGUFValueType] = []


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: npt.NDArray[np.uint32]
    n_elements: int
    n_bytes: int
    data_offset: int
    data: npt.NDArray[Any]
    field: ReaderField


class GGUFReader:
    # I - same as host, S - swapped
    byte_order: Literal['I' | 'S'] = 'I'
    alignment: int = GGUF_DEFAULT_ALIGNMENT

    # Note: Internal helper, API may change.
    gguf_scalar_to_np: dict[GGUFValueType, type[np.generic]] = {
        GGUFValueType.UINT8:   np.uint8,
        GGUFValueType.INT8:    np.int8,
        GGUFValueType.UINT16:  np.uint16,
        GGUFValueType.INT16:   np.int16,
        GGUFValueType.UINT32:  np.uint32,
        GGUFValueType.INT32:   np.int32,
        GGUFValueType.FLOAT32: np.float32,
        GGUFValueType.UINT64:  np.uint64,
        GGUFValueType.INT64:   np.int64,
        GGUFValueType.FLOAT64: np.float64,
        GGUFValueType.BOOL:    np.bool_,
    }

    def __init__(self, path: os.PathLike[str] | str, mode: Literal['r' | 'r+' | 'c'] = 'r'):
        self.data = np.memmap(path, mode = mode)
        offs = 0
        if self._get(offs, np.uint32, override_order = '<')[0] != GGUF_MAGIC:
            raise ValueError('GGUF magic invalid')
        offs += 4
        temp_version = self._get(offs, np.uint32)
        if temp_version[0] & 65535 == 0:
            # If we get 0 here that means it's (probably) a GGUF file created for
            # the opposite byte order of the machine this script is running on.
            self.byte_order = 'S'
            temp_version = temp_version.newbyteorder(self.byte_order)
        version = temp_version[0]
        if version not in READER_SUPPORTED_VERSIONS:
            raise ValueError(f'Sorry, file appears to be version {version} which we cannot handle')
        self.fields: OrderedDict[str, ReaderField] = OrderedDict()
        self.tensors: list[ReaderTensor] = []
        offs += self._push_field(ReaderField(offs, 'GGUF.version', [temp_version], [0], [GGUFValueType.UINT32]))
        temp_counts = self._get(offs, np.uint64, 2)
        offs += self._push_field(ReaderField(offs, 'GGUF.tensor_count', [temp_counts[:1]], [0], [GGUFValueType.UINT64]))
        offs += self._push_field(ReaderField(offs, 'GGUF.kv_count', [temp_counts[1:]], [0], [GGUFValueType.UINT64]))
        tensor_count, kv_count = temp_counts
        offs = self._build_fields(offs, kv_count)
        offs, tensors_fields = self._build_tensors_fields(offs, tensor_count)
        new_align = self.fields.get('general.alignment')
        if new_align is not None:
            if new_align.types != [GGUFValueType.UINT64]:
                raise ValueError('Bad type for general.alignment field')
            self.alignment = new_align.parts[-1][0]
        padding = offs % self.alignment
        if padding != 0:
            offs += self.alignment - padding
        self._build_tensors(offs, tensors_fields)

    _DT = TypeVar('_DT', bound = npt.DTypeLike)

    # Fetch a key/value metadata field by key.
    def get_field(self, key: str) -> Union[ReaderField, None]:
        return self.fields.get(key, None)

    # Fetch a tensor from the list by index.
    def get_tensor(self, idx: int) -> ReaderTensor:
        return self.tensors[idx]

    def _get(
        self, offset: int, dtype: npt.DTypeLike, count: int = 1, override_order: None | Literal['I' | 'S' | '<'] = None,
    ) -> npt.NDArray[Any]:
        count = int(count)
        itemsize = int(np.empty([], dtype = dtype).itemsize)
        end_offs = offset + itemsize * count
        return (
            self.data[offset:end_offs]
            .view(dtype = dtype)[:count]
            .newbyteorder(override_order or self.byte_order)
        )

    def _push_field(self, field: ReaderField, skip_sum: bool = False) -> int:
        if field.name in self.fields:
            raise KeyError(f'Duplicate {field.name} already in list at offset {field.offset}')
        self.fields[field.name] = field
        return 0 if skip_sum else sum(int(part.nbytes) for part in field.parts)

    def _get_str(self, offset: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint8]]:
        slen = self._get(offset, np.uint64)
        return slen, self._get(offset + 8, np.uint8, slen[0])

    def _get_field_parts(
        self, orig_offs: int, raw_type: int,
    ) -> tuple[int, list[npt.NDArray[Any]], list[int], list[GGUFValueType]]:
        offs = orig_offs
        types: list[GGUFValueType] = []
        gtype = GGUFValueType(raw_type)
        types.append(gtype)
        # Handle strings.
        if gtype == GGUFValueType.STRING:
            sparts: list[npt.NDArray[Any]] = list(self._get_str(offs))
            size = sum(int(part.nbytes) for part in sparts)
            return size, sparts, [1], types
        # Check if it's a simple scalar type.
        nptype = self.gguf_scalar_to_np.get(gtype)
        if nptype is not None:
            val = self._get(offs, nptype)
            return int(val.nbytes), [val], [0], types
        # Handle arrays.
        if gtype == GGUFValueType.ARRAY:
            raw_itype = self._get(offs, np.uint32)
            offs += int(raw_itype.nbytes)
            alen = self._get(offs, np.uint64)
            offs += int(alen.nbytes)
            aparts: list[npt.NDArray[Any]] = [raw_itype, alen]
            data_idxs: list[int] = []
            for idx in range(alen[0]):
                curr_size, curr_parts, curr_idxs, curr_types = self._get_field_parts(offs, raw_itype[0])
                if idx == 0:
                    types += curr_types
                idxs_offs = len(aparts)
                aparts += curr_parts
                data_idxs += (idx + idxs_offs for idx in curr_idxs)
                offs += curr_size
            return offs - orig_offs, aparts, data_idxs, types
        # We can't deal with this one.
        raise ValueError('Unknown/unhandled field type {gtype}')

    def _get_tensor(self, orig_offs: int) -> ReaderField:
        offs = orig_offs
        name_len, name_data = self._get_str(offs)
        offs += int(name_len.nbytes + name_data.nbytes)
        n_dims = self._get(offs, np.uint32)
        offs += int(n_dims.nbytes)
        dims = self._get(offs, np.uint64, n_dims[0])
        offs += int(dims.nbytes)
        raw_dtype = self._get(offs, np.uint32)
        offs += int(raw_dtype.nbytes)
        offset_tensor = self._get(offs, np.uint64)
        offs += int(offset_tensor.nbytes)
        return ReaderField(
            orig_offs,
            str(bytes(name_data), encoding = 'utf-8'),
            [name_len, name_data, n_dims, dims, raw_dtype, offset_tensor],
            [1, 3, 4, 5],
        )

    def _build_fields(self, offs: int, count: int) -> int:
        for _ in range(count):
            orig_offs = offs
            kv_klen, kv_kdata = self._get_str(offs)
            offs += int(kv_klen.nbytes + kv_kdata.nbytes)
            raw_kv_type = self._get(offs, np.uint32)
            offs += int(raw_kv_type.nbytes)
            parts: list[npt.NDArray[Any]] = [kv_klen, kv_kdata, raw_kv_type]
            idxs_offs = len(parts)
            field_size, field_parts, field_idxs, field_types = self._get_field_parts(offs, raw_kv_type[0])
            parts += field_parts
            self._push_field(ReaderField(
                orig_offs,
                str(bytes(kv_kdata), encoding = 'utf-8'),
                parts,
                [idx + idxs_offs for idx in field_idxs],
                field_types,
            ), skip_sum = True)
            offs += field_size
        return offs

    def _build_tensors_fields(self, offs: int, count: int) -> tuple[int, list[ReaderField]]:
        tensor_fields = []
        for _ in range(count):
            field = self._get_tensor(offs)
            offs += sum(int(part.nbytes) for part in field.parts)
            tensor_fields.append(field)
        return offs, tensor_fields

    def _build_tensors(self, start_offs: int, fields: list[ReaderField]) -> None:
        tensors = []
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            ggml_type = GGMLQuantizationType(raw_dtype[0])
            n_elems = np.prod(dims)
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offs = int(start_offs + offset_tensor[0])
            item_type: npt.DTypeLike
            if ggml_type == GGMLQuantizationType.F32:
                item_count = n_elems
                item_type = np.float32
            elif ggml_type == GGMLQuantizationType.F16:
                item_count = n_elems
                item_type = np.float16
            else:
                item_count = n_bytes
                item_type = np.uint8
            tensors.append(ReaderTensor(
                name = str(bytes(name_data), encoding = 'utf-8'),
                tensor_type = ggml_type,
                shape = dims,
                n_elements = n_elems,
                n_bytes = n_bytes,
                data_offset = data_offs,
                data = self._get(data_offs, item_type, item_count),
                field = field,
            ))
        self.tensors = tensors


def convert_gguf_to_tokenizer(checkpoint):
    result = GGUFReader(checkpoint)
    # write vocab
    vocab_type = result.fields["tokenizer.ggml.model"]
    vocab_type = str(bytes(vocab_type.parts[vocab_type.data[0]]), encoding = 'utf-8')
    directory = tempfile.mkdtemp()
    if vocab_type == "gpt2":
        # bpe vocab
        merges = result.fields["tokenizer.ggml.merges"]
        with open(os.path.join(directory, "merges.txt"), "w") as temp_file:
            for idx in merges.data:
                data = str(bytes(merges.parts[idx]), encoding = 'utf-8')
                temp_file.write(f"{data}\n")

        tokens = result.fields['tokenizer.ggml.tokens']
        types = result.fields['tokenizer.ggml.token_type']
        vocab_size = len(tokens.data)
        vocab = {}
        special_vocab = {}
        vocab_list = []
        for i, idx in enumerate(tokens.data):
            token = str(bytes(tokens.parts[idx]), encoding='utf-8')
            if token.startswith("[PAD") or token.startswith("<dummy"):
                break
            vocab_list.append(token)
            token_type = int(types.parts[types.data[i]])
            vocab[token] = i
            if token_type == 3:
                special_vocab[i] = {"content": token, "special": True}
        with open(os.path.join(directory, "vocab.json"), "w") as temp_file:
            json.dump(vocab, temp_file, ensure_ascii=False, indent=2)
    else:
        sentencepiece_model_pb2 = import_protobuf()
        vocab = sentencepiece_model_pb2.ModelProto()
        vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
        vocab.trainer_spec.model_type = 2 # BPE
        vocab.trainer_spec.vocab_size = vocab_size
        vocab.trainer_spec.byte_fallback = True
        vocab.normalizer_spec.remove_extra_whitespaces = False
        tokens = result.fields['tokenizer.ggml.tokens']
        scores = result.fields['tokenizer.ggml.scores']
        types = result.fields['tokenizer.ggml.token_type']
        special_vocab = {}
        vocab_list = []
        for i in range(vocab_size):
            new_token = vocab.SentencePiece()
            new_token.piece = str(bytes(tokens.parts[tokens.data[i]]), encoding = 'utf-8')
            if new_token.piece.startswith("[PAD") or new_token.piece.startswith("<dummy"):
                break
            new_token.score = scores.parts[scores.data[i]]
            # llama.cpp tokentype is the same with sentencepiece token type
            new_token.type = int(types.parts[types.data[i]])
            vocab.pieces.append(new_token)
            vocab_list.append(new_token.piece)
            if new_token.type == 3:
                special_vocab[i] = {"content": new_token.piece, "special": True}
        with open(os.path.join(directory, "tokenizer.model"), "wb") as temp_file:
            temp_file.write(vocab.SerializeToString())

    tokenizer_conf = {}
    if 'tokenizer.ggml.bos_token_id' in result.fields:
        tokenizer_conf["bos_token"] = vocab_list[int(result.fields['tokenizer.ggml.bos_token_id'].parts[-1])]
    if 'tokenizer.ggml.eos_token_id' in result.fields:
        tokenizer_conf["eos_token"] = vocab_list[int(result.fields['tokenizer.ggml.eos_token_id'].parts[-1])]
    if 'tokenizer.ggml.padding_token_id' in result.fields:
        tokenizer_conf["pad_token"] = vocab_list[int(result.fields['tokenizer.ggml.padding_token_id'].parts[-1])]
    if 'tokenizer.ggml.unknown_token_id' in result.fields:
        tokenizer_conf["unk_token"] = vocab_list[int(result.fields['tokenizer.ggml.unknown_token_id'].parts[-1])]
    if 'tokenizer.ggml.add_bos_token' in result.fields:
        tokenizer_conf["add_bos_token"] = bool(result.fields['tokenizer.ggml.add_bos_token'].parts[-1])
    if 'tokenizer.ggml.add_eos_token' in result.fields:
        tokenizer_conf["add_eos_token"] = bool(result.fields['tokenizer.ggml.add_eos_token'].parts[-1])
    if special_vocab:
        tokenizer_conf["added_tokens_decoder"] = special_vocab
    with open(os.path.join(directory, "tokenizer_config.json"), "w") as temp_file:
        json.dump(tokenizer_conf, temp_file, indent=2)

    if vocab_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(directory)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(directory)
    shutil.rmtree(directory)
    return tokenizer


def convert_gguf_to_state_dict(checkpoint, config):
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"Cannot find any model weights with `{checkpoint}`")

    result = GGUFReader(checkpoint)
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]), encoding = 'utf-8')
    # write tensor
    head_dim = config.hidden_size // config.num_attention_heads
    kv_dim = config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    tensor_mapping = {
        "token_embd": ("model.embed_tokens", config.vocab_size),
        "output": ("lm_head", config.vocab_size),
        "output_norm": ("model.norm", -1),
        "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm", -1),
        "blk.{bid}.attn_q": ("model.layers.{bid}.self_attn.q_proj", config.hidden_size),
        "blk.{bid}.attn_k": ("model.layers.{bid}.self_attn.k_proj", kv_dim),
        "blk.{bid}.attn_v": ("model.layers.{bid}.self_attn.v_proj", kv_dim),
        "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj", config.hidden_size),
        "blk.{bid}.attn_rot_embd": ("model.layers.{bid}.self_attn.rotary_emb.inv_freq", -1),
        "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm", -1),
        "blk.{bid}.ffn_up": ("model.layers.{bid}.mlp.up_proj", config.intermediate_size),
        "blk.{bid}.ffn_down": ("model.layers.{bid}.mlp.down_proj", config.hidden_size),
        "blk.{bid}.ffn_gate": ("model.layers.{bid}.mlp.gate_proj", config.intermediate_size),
        "blk.{bid}.ffn_up.{xid}": ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w3", config.intermediate_size),
        "blk.{bid}.ffn_down.{xid}": ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w2", config.hidden_size),
        "blk.{bid}.ffn_gate.{xid}": ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w1", config.intermediate_size),
        "blk.{bid}.ffn_gate_inp": ("model.layers.{bid}.block_sparse_moe.gate", config.num_local_experts if hasattr(config, 'num_local_experts') else -1),
    }
    mapping = {}
    # This is how llama.cpp handles name mapping,
    # it's better to use regex match instead though
    max_block_num = 200
    max_expert_num = 8
    for k, v in tensor_mapping.items():
        for i in range(max_block_num):
            for j in range(max_expert_num):
                fk = k.format(bid=i, xid=j)
                fv = v[0].format(bid=i, xid=j)
                if k not in mapping:
                    mapping[fk] = (fv, v[1])

    state_dict = {}
    for ts in result.tensors:
        weight_type = torch.tensor(int(ts.tensor_type), dtype=torch.int)
        layer, suffix = ts.name.rsplit(".", 1)
        new_key, output_dim = mapping[layer]
        new_key += f".{suffix}"
        data = torch.tensor(ts.data)
        if "weight" in ts.name:
            if output_dim != -1:
                data = data.view(output_dim, -1)
            if architecture in ["llama", "internlm2"] and any(
                    k in ts.name for k in ["attn_q", "attn_k"]):
                # change rope style
                data = data.view(output_dim // head_dim, head_dim // 2, 2,-1).permute(
                    0, 2, 1, 3).reshape(output_dim, -1)

            if weight_type > 1:
                state_dict[new_key.replace("weight", "weight_type")] = weight_type
        state_dict[new_key] = data
    return state_dict


def extract_gguf_config(checkpoint):
    result = GGUFReader(checkpoint)
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]), encoding = 'utf-8')
    # Only support llama and qwen2 so far
    if architecture not in ["llama", "qwen2"]:
        raise RuntimeError(f"Unsupported architecture {architecture}")

    # write config
    vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
    context_length = int(result.fields[f'{architecture}.context_length'].parts[-1])
    n_layer = int(result.fields[f'{architecture}.block_count'].parts[-1])
    n_head = int(result.fields[f'{architecture}.attention.head_count'].parts[-1])
    n_local_heads = int(result.fields[f'{architecture}.attention.head_count_kv'].parts[-1])
    intermediate_size = int(result.fields[f'{architecture}.feed_forward_length'].parts[-1])
    norm_eps = float(result.fields[f'{architecture}.attention.layer_norm_rms_epsilon'].parts[-1])
    dim = int(result.fields[f'{architecture}.embedding_length'].parts[-1])
    if 'tokenizer.ggml.bos_token_id' in result.fields:
        bos_token_id = int(result.fields['tokenizer.ggml.bos_token_id'].parts[-1])
    else:
        bos_token_id = 1
    if 'tokenizer.ggml.eos_token_id' in result.fields:
        eos_token_id = int(result.fields['tokenizer.ggml.eos_token_id'].parts[-1])
    else:
        eos_token_id = 2
    arch = "MixtralForCausalLM"
    if architecture == "qwen2":
        arch = "Qwen2ForCausalLM"
        name = "qwen2"
    elif f'{architecture}.expert_count' in result.fields:
        arch = "MixtralForCausalLM"
        name = "mixtral"
    else:
        arch = "LlamaForCausalLM"
        name = "llama"
    model_config= {
        "architectures": [arch],
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
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
    if f'{architecture}.rope.freq_base' in result.fields:
        model_config['rope_theta'] = float(result.fields[f'{architecture}.rope.freq_base'].parts[-1])
    if f'{architecture}.expert_count' in result.fields:
        model_config['num_local_experts'] = int(result.fields[f'{architecture}.expert_count'].parts[-1])
        model_config['num_experts_per_tok'] = int(result.fields[f'{architecture}.expert_used_count'].parts[-1])
    config_class = CONFIG_MAPPING[name]
    hf_config = config_class.from_dict(model_config)
    return hf_config
