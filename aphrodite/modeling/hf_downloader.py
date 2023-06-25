"""HuggingFace model downloader, uses `aria2c` by default"""
import fileblock
import glob
import json
import os
from typing import Iterator, List, Optional, Tuple
from huggingface_hub import snapshot_download
import numpy as np
import torch
from tqdm.auto import tqdm
import shutil
import subprocess

class Disabledtqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)

def download_with_aria2(url: str, output_path: str) -> None:
    cmd = ['aria2c', '-x', '16', '-s', '16', '-d', os.path.dirname(output_path), '-o', os.path.basename(output_path), url]
    subprocess.run(cmd)

def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_np_cache: bool = False,
    use_aria2: bool = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    # We'll perform a file lock directory here so multiple processes won't download the same file.
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))

    # Check if `aria2` is installed and use it for downloading if `use_aria2` is enabled.
    use_aria2_download = use_aria2 and shutil.which('aria2c') is not None

    # Here's the HF downloader
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        with lock:
            hf_folder = snapshot_download(model_name_or_path, allow_patterns="*.bin", cache_dir=cache_dir, tqdm_class=Disabledtqdm, use_aria2_download=use_aria2_download)
    else:
        hf_folder = model_name_or_path

    hf_bin_files = glob.glob(os.path.join(hf_folder, "*.bin"))

    if use_np_cache:
        np_folder = os.path.join(hf_folder, 'np')
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, 'weight_names.json')
        with lock:
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_bin_files:
                    state = torch.load(bin_file, map_location="cpu")  # loaded on CPU, make sure you have enough DRAM
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, 'w') as f:
                    json.dump(weight_names, f)
        
        with open(weight_names_file, 'r') as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)

    else:
        for bin_file in hf_bin_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param

def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weights: torch.Tensor,
    param_name: str,
    column_parallel_weight_names: List[str],
    row_parallel_weight_names: List[str],
    tensor_model_parallel_rank: int,
) -> None:
    for p in column_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[0]
            loaded_weight = loaded_weights[
                shard_size * tensor_model_parallel_rank
                :shard_size * (tensor_model_parallel_rank + 1)]
            break
    for p in row_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[1]
            loaded_weight = loaded_weight[
                :,
                shard_size * tensor_model_parallel_rank
                :shard_size * (tensor_model_parallel_rank + 1)]
            break
    assert param.shape == loaded_weight.shape
    param.data.copy_(loaded_weights)


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance measurements.
    Additionally, the model weights should not cause NaNs in the forward pass. The given values
    above apparently work well for most models.
    """
    for param in model.state_dict().values():
        param.data.uniform_(low, high)
