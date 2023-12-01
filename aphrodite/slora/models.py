import copy
import json
import logging
import math
import os
import re
from typing import (
    Any, Callable, Dict, Hashable, List, Optional, Tuple, Type,
    Union)

import safetensors.torch
import torch
from torch import nn

from aphrodite.common.config import LoRAConfig
from aphrodite.common.utils import LRUCache

from aphrodite.slora.layer import (LoRALayer, LoRAMapping, from_layer,
                                   from_layer_sampler)
from aphrodite.slora.lora import LoRA
from aphrodite.slora.utils import parse_fine_tuned_lora_name, replace_submodule

logger = logging.getLogger(__name__)

PACKED_MODULES_CFG = {
    "qkv_proj": [
        "q_proj",
        "k_proj",
        "v_proj",
    ],
    "gate_up_proj": [
        "gate_proj",
        "up_proj",
    ],
}

TARGET_MODULES_QKV = [
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]

EMBEDDING_MODULES = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}

EMBEDDING_PADDING_MODULES = ["lm_head"]

_GLOBAL_LORA_ID = 0


def convert_mapping(
        mapping: LoRAMapping, lora_id_to_index: List[Optional[int]],
        max_loras: int, vocab_size: int, extra_vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts LoRAMapping to index tensors.
    
    Args:
        mapping: LoRAMapping rows in a batch to LoRA ids.
        lora_id_to_index: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows
                to LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests
                to LoRA indices for sampler. For generation, this will be
                the same as base_indicies. For prefill, this will map
                requests to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with pading. Same as
                sampler_indicies, but -1 is replaced with max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a 
                embeddings.
            indices_len: List of lengths of the above tensors.
    """
    indices = list(mapping.index_mapping).copy()
    embedding_indices = indices.copy()
    lora_indices = indices.copy()
    prompt_mapping = [
        lora_id_to_index.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(indices)):
        # TODO: Index can be slow, optimize.
        lora_idx = (lora_id_to_index.index(indices[i])
                    if indices[i] > 0 else -1)
        embedding_indices[i] = lora_idx if indices[i] > 0 else 0
        indices[i] = i
        lora_indices[i] = lora_idx

    indices = torch.tensor([indices, lora_indices, embedding_indices],
                           dtype=torch.long,
                           device="cuda")
    prompt_mapping = torch.tensor(prompt_mapping,
                                  device="cuda",
                                  dtype=torch.long)
    embeddings_indices = torch.stack([
        indices[2] * extra_vocab_size,
        indices[2] * (vocab_size + extra_vocab_size)
    ])
    embeddings_indices[embeddings_indices == -1] = max_loras -1
    base_indices = indices[1]
    sampler_indices = prompt_mapping
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_loras - 1
    sampler_indices_padded = (
        torch.arange(
            0, len(sampler_indices_padded), device="cuda", dtype=torch.long) +
        (sampler_indices_padded * len(sampler_indices_padded)))
    indices_len = (base_indices.shape[-1], sampler_indices.shape[-1],
                   sampler_indices_padded.shape[-1],
                   embeddings_indices.shape[-1])

    return (base_indices, sampler_indices, sampler_indices_padded,
            embeddings_indices, indices_len)

def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID

def _create_dummy_lora(module_name: str,
                       input_dim: int,
                       output_dim: int,
                       rank: int,
                       dtype: torch.dtype,
                       device: torch.device,
                       embeddings_tensor_dim: Optional[int] = None) -> "LoRA":
    lora_a = torch.zeros([input_dim, rank], dtype=dtype, device=device)
    lora_b = torch.zeros([rank, output_dim], dtype=dtype, device=device)
    embeddings_tensor = torch.rand(
        10, embeddings_tensor_dim, dtype=dtype,
        device=device) if embeddings_tensor_dim else None
    if str(device) == "cpu":
        lora_a = lora_a.pin_memory()
        lora_b = lora_b.pin_memory()
        if embeddings_tensor is not None:
            embeddings_tensor = embeddings_tensor.pin_memory()
    return LoRA(
        module_name,
        rank=rank,
        lora_alpha=1,
        lora_a=lora_a,
        lora_b=lora_b,
        embeddings_tensor=embeddings_tensor,
    )

class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(
            self,
            lora_model_id: int,
            rank: int,
            loras: Dict[str, LoRA],
    ) -> None:
        self.id = lora_model_id
        assert (lora_model_id >
                0), f"a valid LoRA ID should be greater than 0, got {self.id}"
        self.rank = rank
        self.loras: Dict[str, LoRA] = loras
                         
    def get_lora(self, module_name: str) -> Optional[LoRA]:
        """Get LoRA for a given module by name."""
        return self.loras.get(module_name, None)

    
    # TODO: see if we can derive target_embedding_padding automatically
    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        rank: int,
        lora_alpha: int,
        tensors: Dict[str, torch.Tensor],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        target_embedding_padding: Optional[int] = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a dict of tensors."""
        loras: Dict[str, LoRA] = {}
        for tensor_name, tensor in tensors.items():
            module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name)
            if module_name not in loras:
                lora_embeddings_tensor = None
                if embeddings:
                    embeddings_module = next(
                        (k for k in EMBEDDING_MODULES if k in module_name),
                        None)
                    if embeddings_module:
                        lora_embeddings_tensor = embeddings[
                            EMBEDDING_MODULES[embeddings_module]].to(
                                device=device, dtype=dtype)
                        if device == "cpu":
                            lora_embeddings_tensor = (
                                lora_embeddings_tensor.pin_memory())
                loras[module_name] = LoRA(module_name, rank, lora_alpha, None,
                                          None, lora_embeddings_tensor)
            if is_lora_a:
                loras[module_name].lora_a = tensor.to(device=device,
                                                      dtype=dtype).t()
                if device == "cpu":
                    loras[module_name].lora_a = (
                        loras[module_name].lora_a.pin_memory())
            else:
                loras[module_name].lora_b = tensor.to(device=device,
                                                      dtype=dtype).t()
                if any(name in module_name
                       for name in EMBEDDING_PADDING_MODULES
                       ) and target_embedding_padding is not None:
                    lora_b = loras[module_name].lora_b
                    assert target_embedding_padding >= lora_b.shape[1]
                    addition = target_embedding_padding - lora_b.shape[1]
                    loras[module_name].lora_b = torch.nn.functional.pad(
                        lora_b, (0, addition))
                    if device == "cpu":
                        loras[module_name].lora_b = (
                            loras[module_name].lora_b.pin_memory())

        for _, lora in loras.items():
            lora.optimize()
        return cls(lora_model_id, rank, loras)

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        lora_model_id: Optional[int] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        target_embedding_padding: Optional[int] = None) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint."""
        lora_config_path = os.path.join(lora_dir, "adapter_config.json")
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        new_embeddings_tensor_path = os.path.join(
            lora_dir, "new_embeddings.safetensors")
        new_embeddings_bin_file_path = os.path.join(lora_dir,
                                                    "new_embeddings.bin")
        if os.path.isfile(lora_tensor_path):
            tensors = safetensors.torch.load_file(lora_tensor_path)
        elif os.path.isfile(lora_bin_file_path):
            tensors = torch.load(lora_bin_file_path)
        else:
            raise ValueError(f"{lora_dir} doesn't contain LoRA adapters.")

        embeddings = None
        if os.path.isfile(new_embeddings_tensor_path):
            embeddings = safetensors.torch.load_file(
                new_embeddings_tensor_path)
        elif os.path.isfile(new_embeddings_bin_file_path):
            embeddings = torch.load(new_embeddings_bin_file_path)

        with open(lora_config_path, "r") as f:
            config = json.load(f)
        rank = config["r"]
        lora_alpha = config["lora_alpha"]
        return cls.from_lora_tensors(
            lora_model_id=get_lora_id()
            if lora_model_id is None else lora_model_id,
            rank=rank,
            lora_alpha=lora_alpha,
            tensors=tensors,
            device=device,
            dtype=dtype,
            embeddings=embeddings,
            target_embedding_padding=target_embedding_padding,
        )
