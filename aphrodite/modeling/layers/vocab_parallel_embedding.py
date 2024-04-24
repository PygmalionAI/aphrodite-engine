from typing import Optional, Sequence

import torch
from torch.nn.parameter import Parameter

from aphrodite.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from aphrodite.modeling.layers.linear import UnquantizedLinearMethod
from aphrodite.modeling.utils import set_weight_attrs

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int,
                   pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int,
                                              rank: int) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int,
                                       world_size: int) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                     rank)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None,
                 linear_method=None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 is_input_emb: bool = True):
        super().__init__()

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.num_embeddings_padded = pad_vocab_size(num_embeddings,
                                                    padding_size)
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.tp_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = (
            vocab_range_from_global_vocab_size(
                self.num_embeddings_padded, get_tensor_model_parallel_rank(),
                self.tp_size))
        self.num_embeddings_per_partition = (self.vocab_end_index -
                                             self.vocab_start_index)
        idx = 0 if is_input_emb else 1
        if linear_method is None or not linear_method.quant_config.quant_vocab(
        )[idx]:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(
            self.embedding_dim, [self.num_embeddings_per_partition],
            self.embedding_dim, self.num_embeddings_padded, params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.nn.parameter.Parameter):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)
        if output_dim is not None:
            shard_offset = self.vocab_start_index
            shard_size = min(self.vocab_end_index,
                             self.org_vocab_size) - shard_offset
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
            loaded_weight = loaded_weight.narrow(output_dim, shard_offset,
                                                 shard_size)
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            vocab_shape = list(loaded_weight.shape)
            if output_dim is not None:
                if packed_dim == output_dim:
                    vocab_shape[output_dim] = (
                        self.num_embeddings_per_partition // param.pack_factor)
                else:
                    vocab_shape[output_dim] = self.num_embeddings_per_partition
            param.materialize(vocab_shape, dtype=loaded_weight.dtype)
        if output_dim is not None:
            param.data.narrow(
                output_dim, 0,
                loaded_weight.shape[output_dim]).copy_(loaded_weight)
        else:
            param.data.copy_(loaded_weight)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            input_mask = ((input_ < self.vocab_start_index) |
                          (input_ >= self.vocab_end_index))
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = self.linear_method.apply_embedding(
            self.linear_weights, masked_input)
        # output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 bias: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 linear_method=None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__(num_embeddings, embedding_dim, params_dtype,
                         linear_method, org_num_embeddings, padding_size,
                         False)
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader
            })
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        logits = self.linear_method.apply_weights(self.linear_weights, input_)
        if self.bias is not None:
            logits += self.bias
        return logits


class ParallelTWEHead(torch.nn.Module):
    """Parallelized tie word embeddings head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are read from a VocabParallelEmbedding.

    Args:
        embeddings: the VocabParallelEmbedding to mirror
    """

    def __init__(self, embeddings: VocabParallelEmbedding):
        super().__init__()
        self.linear_method = embeddings.linear_method
        self.linear_weights = embeddings.linear_weights

    def forward(self, input_):
        logits = self.linear_method.apply_weights(self.linear_weights, input_)
        return logits
