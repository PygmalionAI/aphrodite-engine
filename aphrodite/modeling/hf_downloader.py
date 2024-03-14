"""Utilities for downloading and initializing model weights."""
import filelock
import glob
import fnmatch
import json
import os
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Tuple
from loguru import logger

import gguf
from huggingface_hub import snapshot_download, HfFileSystem
import numpy as np
from safetensors.torch import load_file, save_file, safe_open
import torch
from transformers import PretrainedConfig
from tqdm.auto import tqdm

from aphrodite.common.config import ModelConfig
from aphrodite.common.logger import get_loading_progress_bar
from aphrodite.modeling.layers.quantization import (get_quantization_config,
                                                    QuantizationConfig)


class Disabledtqdm(tqdm):  # pylint: disable=inconsistent-mro

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


# TODO: Move this to another place.
def get_quant_config(model_config: ModelConfig) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)
    # Read the quantization config from the HF model config, if available.
    # if the quantization if "gguf", we skip and return quant_cls()
    if model_config.quantization in ["exl2", "gguf"]:
        return quant_cls()
    hf_quant_config = getattr(model_config.hf_config, "quantization_config",
                              None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, model_config.download_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          revision=model_config.revision,
                                          allow_patterns="*.json",
                                          cache_dir=model_config.download_dir,
                                          tqdm_class=Disabledtqdm)
    else:
        hf_folder = model_name_or_path
    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(
            f.endswith(x) for x in quant_cls.get_config_filenames())
    ]
    if len(quant_config_files) == 0:
        raise ValueError(
            f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}")

    quant_config_file = quant_config_files[0]
    with open(quant_config_file, "r") as f:
        config = json.load(f)
    return quant_cls.from_config(config)


def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npcache":
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

        logger.info(f"Downloading model weights {allow_patterns}")
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm,
                                          revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
            "trainer_state.json",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def convert_gguf_to_state_dict(checkpoint, config):
    if not os.path.isfile(checkpoint):
        raise RuntimeError(
            f"Cannot find any model weights with `{checkpoint}`")

    result = gguf.GGUFReader(checkpoint)
    # write tensor
    kv_dim = config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    tensor_mapping = {
        "token_embd": ("model.embed_tokens", config.vocab_size),
        "output": ("lm_head", config.vocab_size),
        "output_norm": ("model.norm", -1),
        "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm", -1),
        "blk.{bid}.attn_q": ("model.layers.{bid}.self_attn.q_proj",
                             config.hidden_size),
        "blk.{bid}.attn_k": ("model.layers.{bid}.self_attn.k_proj", kv_dim),
        "blk.{bid}.attn_v": ("model.layers.{bid}.self_attn.v_proj", kv_dim),
        "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj",
                                  config.hidden_size),
        "blk.{bid}.attn_rot_embd":
        ("model.layers.{bid}.self_attn.rotary_emb.inv_freq", -1),
        "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm",
                               -1),
        "blk.{bid}.ffn_up": ("model.layers.{bid}.mlp.up_proj",
                             config.intermediate_size),
        "blk.{bid}.ffn_down": ("model.layers.{bid}.mlp.down_proj",
                               config.hidden_size),
        "blk.{bid}.ffn_gate": ("model.layers.{bid}.mlp.gate_proj",
                               config.intermediate_size),
        "blk.{bid}.ffn_up.{xid}":
        ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w3",
         config.intermediate_size),
        "blk.{bid}.ffn_down.{xid}":
        ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w2",
         config.hidden_size),
        "blk.{bid}.ffn_gate.{xid}":
        ("model.layers.{bid}.block_sparse_moe.experts.{xid}.w1",
         config.intermediate_size),
        "blk.{bid}.ffn_gate_inp": ("model.layers.{bid}.block_sparse_moe.gate",
                                   config.num_local_experts if hasattr(
                                       config, "num_local_experts") else -1),
    }
    mapping = {}
    # This is how llama.cpp handles name mapping,
    # it's better to use regex match instead doe
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
    with get_loading_progress_bar() as progress:
        task = progress.add_task("[cyan]Converting GGUF tensors to PyTorch...",
                                 total=len(result.tensors))
        for ts in result.tensors:
            weight_type = torch.tensor(int(ts.tensor_type), dtype=torch.int)
            layer, suffix = ts.name.rsplit(".", 1)
            new_key, output_dim = mapping[layer]
            new_key += f".{suffix}"
            data = torch.tensor(ts.data)
            if output_dim != -1:
                data = data.view(output_dim, -1)
            if weight_type > 1:
                state_dict[new_key.replace("weight",
                                           "weight_type")] = weight_type
            state_dict[new_key] = data
            progress.update(task, advance=1)
    return state_dict


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    config: Optional[PretrainedConfig] = None,
    fall_back_to_pt: Optional[bool] = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    if model_name_or_path.endswith("gguf"):
        for name, param in convert_gguf_to_state_dict(model_name_or_path,
                                                      config).items():
            yield name, param
        return

    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        load_format=load_format,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)

    if load_format == "npcache":
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    if isinstance(param, torch.nn.parameter.UninitializedParameter):
        param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)
    assert param.size() == loaded_weight.size()
    param.data.copy_(loaded_weight)


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            param.data.uniform_(low, high)
