from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import GPTJConfig

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.layers.activation import get_act_fn
from aphrodite.modeling.layers.attention import PagedAttentionWithRoPE
from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.hf_downloader import hf_model_weights_iterator, load_tensor_parallel_weights
from aphrodite.modeling.megatron.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from aphrodite.modeling.megatron.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from aphrodite.common.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]

class GPTJAttention(nn.Module):
