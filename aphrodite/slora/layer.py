from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from aphrodite.config import LoRAConfig
from aphrodite.lora.slora import add_lora, add_lora_slice, bgmv
from aphrodite.modeling.megatron.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce
)
from aphrodite.modeling.layers.linear import (ColumnParallelLinear,
                                              RowParallelLinear,
                                              QKVParallelLinear,
                                              MergedColumnParallelLinear)
from aphrodite.modeling.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from aphrodite.modeling.megatron.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from aphrodite.modeling.megatron.utils import split_tensor_along_last_dim

if TYPE_CHECKING:
    pass

def _apply_lora(
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
):
    """"Applies LoRA to each input.
    
    This method applies all LoRAs to each input. It uses the
    indices vector to determine which LoRA yields the correct
    output. An index of -1 means no LoRA should be applied.
    This method adds the final LoRA results to the output.

    Input shapes:
        x: (batch_size, hidden_dim)
        lora_a_stacked: (num_loras, lora_rank, hidden_dim)
        lora_b_stacked: (num_loras, output_dim, lora_rank)
        indices: (batch_size)
        output: (batch_size, output_dim)
    """
    org_output = output
    if x.ndim == 3:
        x = x.view(x.shape[0] * x.shape[1], -1)
    if output.ndim == 3:
        output = output.view(output.shape[0] * output.shape[1], -1)
    add_lora(output, x, lora_a_stacked, lora_b_stacked, indices, 0, 1.0)
    return output.view_as(org_output)


def _apply_lora_packed_2slice(
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, torch.Tensor],
        lora_b_stacked: Tuple[torch.Tensor, torch.Tensor],
        indices: torch.Tensor,
        output: torch.Tensor,
        output_dim: int,
):
    """"Apply LoRA to each input.
    
    This method applies all LoRAs to each input. It uses the
    indices vector to determine which LoRA yields the correct
    output. An index of -1 means no LoRA should be applied.
    This method adds the final LoRA results to the output.

    This method is used for layers that are composed of 2 sublayers
    (slices) packed together (e.g. gate_proj + up_proj ->
    gate_up_proj).

    Both slices must have the same size (output_dim), meaning the
    output tensor will have size output_dim*2.

    Input shapes:
        x: (batch_size, hidden_dim)
        lora_a_stacked: 2-tuple of (num_loras, lora_rank, hidden_dim)
        lora_b_stacked: 2-tuple of (num_loras, output_dim, lora_rank)
        indices: (batch_size)
        output: (batch_size, output_dim*2)
        output_dim: scalar
    """

    org_output = output
    if x.ndim == 3:
        x = x.view(x.shape[0] * x.shape[1], -1)
    if output.ndim == 3:
        output = output.view(output.shape[0] * output.shape[1], -1)
    add_lora_slice(output, x, lora_a_stacked[0], lora_b_stacked[0], indices, 0,
                   1.0, 0, output_dim)
    add_lora_slice(output, x, lora_a_stacked[1], lora_b_stacked[1], indices, 0,
                   1.0, output_dim, output_dim)
    return output.view_as(org_output)


def _apply_lora_packed_3slice(
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        lora_b_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        indices: torch.Tensor,
        output: torch.Tensor,
        output_slices: Tuple[int, int],
):
    """Applies LoRA to each input.
    
    This method applies all LoRAs to each input. It uses the
    indices vector to determine which LoRA yields the correct
    output. An index of -1 means no LoRA should be applied.
    This method adds the final LoRA results to the output.

    This method is used for layers that are composed of 3 sublayers
    (slices) packed together (attention projection). The first slice
    (Q) may have different size from the two subsequent slices (K, V).

    Input shapes:
        x: (batch_size, hidden_dim)
        lora_a_stacked: 3-tuple of (num_loras, lora_rank, hidden_dim)
        lora_b_stacked: 3-tuple of (num_loras, output_dim, lora_rank)
        indices: (batch_size)
        output: (batch_size, q_slice_size + 2*kv_slice_size)
        output_slices: 2-tuple of (q_slice_size, kv_slice_size)
    """"
    org_output = output
    if x.ndim == 3:
        x = x.view(x.shape[0] * x.shape[1], -1)
    if output.ndim == 3:
        output = output.view(output.shape[0] * output.shape[1], -1)
    add_lora_slice(output, x, lora_a_stacked[0], lora_b_stacked[0], indices, 0,
                   1.0, 0, output_slices[0])
    add_lora_slice(output, x, lora_a_stacked[1], lora_b_stacked[1], indices, 0,
                     1.0, output_slices[0], output_slices[1])
    add_lora_slice(output, x, lora_a_stacked[2], lora_b_stacked[2], indices, 0,
                   1.0, output_slices[0] + output_slices[1], output_slices[1])
    return output.view_as(org_output)


@dataclass
class LoRAMapping:
    index_mapping: Tuple[int, ...]
    prompt_mapping: Tuple[int, ...]

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, self.__class__)
                and self.prompt_mapping == __value.prompt_mapping
                and self.index_mapping == __value.index_mapping)
    
    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)


class LoRALayer(nn.Module):

    def create_lora_weights(self, max_loras: int, lora_config: LoRAConfig,
                            model_config: PretrainedConfig) -> None:
        """Initializes the LoRA matrices."""
        ...
    
    def reset_lora(self, index: int):
        """Resets the LoRA weights at index back to 0."""
        ...
    
    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        """Overwrites LoRA tensors at index."""
        ...

    def set_mapping(
            self,
            base_indices: torch.Tensor,
            sampler_indices: torch.Tensor,
            sampler_indices_padded: torch.Tensor,
            embeddings_indices: torch.Tensor,
            indices_len: List[int],
    ):
        """Sets the mapping indices."""
        ...
    

class LoRAVocabParallelEmbedding(LoRALayer):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:

        lora_vocab_start_idx = self.base_layer.org_vocab_size
        weights_idx = None
        if self.base_layer.vocab_end_index > lora_vocab_start_idx:
            weights_idx = max(
                lora_vocab_start_idx - self.base_layer.vocab_start_index, 0)
            self.embeddings_slice = (self.base_layer.vocab_start_index - 
                                     self.base_layer.org_vocab_size +
                                     weights_idx,
                                     self.base_layer.vocab_end_index -
                                     self.base_layer.org_vocab_size)
            self.embeddings_weights = self.base_layer.weight.data[weights_idx:]
            self.embeddings_weights.fill_(0)
        else:
            self.embeddings_slice = None
            self.embeddings_weights = None
        
        self.embeddings_tensors = torch.zeros(
            (
                max_loras,
                lora_config.lora_extra_vocab_size,
                self.base_layer.embedding_dim,
            ),
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.org_vocab_size +
                lora_config.lora_extra_vocab_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                self.base_layer.embedding_dim,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked_2d = self.lora_a_stacked.view(
            self.lora_a_stacked.shape[0] * self.lora_a_stacked.shape[1],
            self.lora_a_stacked.shape[2],
        )
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.embeddings_indices = None
    
    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        self.lora_a_stacked[index, :lora_a.shape[0], :lora_a.shape[1]].copy_(
            lora_a, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensor[
                index, :embeddings_tensor.shape[0], :embeddings_tensor.
                shape[1]].copy_(embeddings_tensor, non_blocking=True)
            if self.embeddings_slice is not None:
                self.embeddings_weights.copy_(
                    self.embeddings_tensors.view(
                        self.embeddings_tensors.shape[0] *
                        self.embeddings_tensors.shape[1],
                        self.embeddings_tensors.shape[2])
                    [self.embeddings_slice[0]:self.embeddings_slice[1]])
    
    def set_mapping(
            self,
            base_indices: torch.Tensor,
            sampler_indices: torch.Tensor,
            sampler_indices_padded: torch.Tensor,
            embeddings_indices: torch.Tensor,
            indices_len: List[int],
    ):
        self.indices = base_indices
        self.embeddings_indices = embeddings_indices
        self.indices_len = indices_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = x > self.base_layer.org_vocab_size - 1
        indices = self.embeddings_indices[1][:self.indices_len[3]].view_as(x)
        full_lora_a_embeddings = F.embedding(
            x + indices,
            self.lora_a_stacked_2d,
        )
        indices = self.embeddings_indices[0][:self.indices_len[3]].view_as(x)
        full_output = self.base_layer.forward(
            x.add_(indices * added_tokens_mask))
        
        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(
                full_output.shape[0] * full_output.shape[1], 1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] *
                full_lora_a_embeddings.shape[1], -1)
        bgmv(full_output, full_lora_a_embeddings, self.lora_b_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        return full_output.view_as(full_output_org)


class LoRAColumnParallelLinear(LoRALayer):

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_a_stacked = torch.zeros(
            max_loras,
            1,
            lora_config.max_lora_rank,
            self.base_layer.weight.shape[1],
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            max_loras,
            1,
            self.base_layer.weight.shape[0],
            lora_config.max_lora_rank,
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def set_lora(
            self,
            index: int,
            lora_a: torch.Tensor,
            lora_b: torch.Tensor,
            embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        _apply_lora(
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[0]],
            output,
        )
        return output
    
    def forward(self, input_):
        """Forward of ColumnParallelLinear.
        
        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)
        
        # matmul
        output_parallel = self.apply_weights(input_, bias)
        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @property
    def linear_weights(self):
        return self.base_layer.linear_weights
             
class LoRAMergedColumnParallelLinear2Slice(LoRAColumnParallelLinear):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (e.g. gate_proj + up_proj -> gate_up_proj).
    
    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)
    
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None) -> None:
        n_slices = 2
        if not (len(self.base_layer.output_sizes) == n_slices
                and self.base_layer.output_sizes[0]
                == self.base_layer.output_sizes[1]):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size.")
        self.tp_size = get_tensor_model_parallel_world_size()

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ) for _ in range(n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.base_layer.weight.shape[0] // 2,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ) for _ in range(n_slices))

        self.indices: Optional[torch.Tensor] = None
        self.output_dim = self.lora_b_stacked[0].shape[2]

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_b_stacked[1][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_dim
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_b = lora_b[0][:,
                               start_idx:end_idx], lora_b[1][:,
                                                             start_idx:end_idx]
        
        if lora_a[0] is not None:
            self.lora_a_stacked[0][
                index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(
                    lora_a[0].T, non_blocking=True)
            self.lora_b_stacked[0][
                index, 0, :lora_b[0].shape[1], :lora_b[0].shape[0]].copy_(
                    lora_b[0].T, non_blocking=True)
        if lora_a[1] is not None:
            self.lora_a_stacked[1][
                index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(
                    lora_a[1].T, non_blocking=True)
            self.lora_b_stacked[1][
                index, 0, :lora_b[1].shape[1], :lora_b[1].shape[0]].copy_(
                    lora_b[1].T, non_blocking=True)
    
    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        _apply_lora_packed_2slice(
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[0]],
            output,
            self.output_din,
        )
        return output