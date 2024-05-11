"""Utilities for downloading and initializing model weights."""
import fnmatch
import glob
import json
import os
from collections import defaultdict
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import filelock
import huggingface_hub.constants
import numpy as np
import torch
from huggingface_hub import HfFileSystem, snapshot_download
from loguru import logger
from safetensors.torch import load_file, safe_open, save_file
from tqdm.auto import tqdm
from transformers import PretrainedConfig, AutoModelForCausalLM

from aphrodite.common.config import ModelConfig
from aphrodite.quantization.gguf_utils import (GGUFReader, get_tensor_name_map,
                                               MODEL_ARCH_NAMES)
from aphrodite.common.logger import get_loading_progress_bar
from aphrodite.quantization import (QuantizationConfig,
                                    get_quantization_config)
from aphrodite.quantization.schema import QuantParamSchema

_xdg_cache_home = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
_aphrodite_filelocks_path = os.path.join(_xdg_cache_home, "aphrodite/locks/")


def enable_hf_transfer():
    """automatically activates hf_transfer
    """
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa
            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


class Disabledtqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else _aphrodite_filelocks_path
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.SoftFileLock(os.path.join(lock_dir, lock_file_name))
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
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=model_config.download_dir,
                tqdm_class=Disabledtqdm,
            )
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

        logger.info(f"Using model weights format {allow_patterns}")
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                tqdm_class=Disabledtqdm,
                revision=revision,
            )
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
            "hidden_states.safetensors",  # exllamav2
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
    model_type = config.model_type
    # hack: ggufs have a different name than transformers
    if model_type == "cohere":
        model_type = "command-r"
    arch = None
    for key, value in MODEL_ARCH_NAMES.items():
        if value == model_type:
            arch = key
            break
    if arch is None:
        raise RuntimeError(f"Unknown model_type: {model_type}")
    num_layers = config.num_hidden_layers
    name_map = get_tensor_name_map(arch, num_layers)
    with torch.device("meta"):
        dummy_model = AutoModelForCausalLM.from_config(config)
    state_dict = dummy_model.state_dict()

    gguf_to_hf_name_map = {}
    keys_to_remove = []
    for hf_name in state_dict:
        name, suffix = hf_name.rsplit(".", 1)
        gguf_name = name_map.get_name(name)
        if gguf_name:
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
        elif name == "lm_head":
            keys_to_remove.append(hf_name)
            logger.warning(
                f"GGUF tensor name for {hf_name} not found, "
                "this is normal if the model uses tie word embeddings.")
        else:
            logger.warning(
                f"GGUF tensor name for {hf_name} in hf state_dict not found.")
    for key in keys_to_remove:
        state_dict.pop(key)

    if os.path.isfile(checkpoint):
        results = [GGUFReader(checkpoint)]
    elif os.path.isdir(checkpoint):
        results = [
            GGUFReader(os.path.join(checkpoint, file))
            for file in os.listdir(checkpoint)
            if os.path.splitext(file)[-1].lower() == ".gguf"
        ]
    else:
        raise RuntimeError(
            f"Cannot find any model weights with `{checkpoint}`")

    with get_loading_progress_bar() as progress:
        task = progress.add_task(
            "[cyan]Converting GGUF tensors to PyTorch...",
            total=sum([len(result.tensors) for result in results]),
        )
        for result in results:
            for ts in result.tensors:
                try:
                    hf_name = gguf_to_hf_name_map[ts.name]
                except KeyError:
                    logger.warning(
                        f"hf tensor name for {ts.name} in GGUF not found.")
                    continue
                data = torch.tensor(ts.data)
                if state_dict[hf_name].dim() == 2:
                    data = data.view(state_dict[hf_name].shape[0], -1)
                state_dict[hf_name] = data
                weight_type = torch.tensor(int(ts.tensor_type),
                                           dtype=torch.int)
                if weight_type > 1:
                    state_dict[hf_name.replace("weight",
                                               "weight_type")] = weight_type
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
        revision=revision,
    )

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


def kv_cache_scales_loader(
        filename: str, tp_rank: int, tp_size: int, num_hidden_layers: int,
        model_type: Optional[str]) -> Iterable[Tuple[int, float]]:
    """
    A simple utility to read in KV cache scaling factors that have been
    previously serialized to disk. Used by the model to populate the appropriate
    KV cache scaling factors. The serialization should represent a dictionary
    whose keys are the TP ranks and values are another dictionary mapping layers
    to their KV cache scaling factors.
    Keep this function in sync with the output of examples/fp8/extract_scales.py
    """
    try:
        with open(filename) as f:
            context = {
                "model_type": model_type,
                "num_hidden_layers": num_hidden_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
            schema_dct = json.load(f)
            schema = QuantParamSchema.model_validate(schema_dct,
                                                     context=context)
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]
            return layer_scales_map.items()

    except FileNotFoundError:
        logger.error(f"File or directory '{filename}' not found.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in file '{filename}'.")
    except Exception as e:
        logger.error(f"An error occurred while reading '{filename}': {e}")
    # This section is reached if and only if any of the excepts are hit
    # Return an empty iterable (list) => no KV cache scales are loaded
    # which ultimately defaults to 1.0 scales
    logger.warning("Defaulting to KV cache scaling factors = 1.0 "
                   f"for all layers in TP rank {tp_rank} "
                   "as an error occurred during loading.")
    return []


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
