"""Configuration"""
from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig

from aphrodite.common.logger import init_logger
from aphrodite.common.utils import get_cpu_memory

logger = init_logger(__name__)

_GiB = 1 << 30

class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the HF model to use.
        tokenizer: Name or path of the HF tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        download_dir: Directory to download and load the weights, defaults to
            default HF cache directory.
        use_np_weights: Save a numpy copy of model weights for faster loading.
            This can increase the disk usage by up to 2x, and the model will be
            loaded into CPU memory first.
        use_dummy_weights: Use dummy values for model weights (for profiling).
        dtype: Datatype for model weights and activations. The "auto" option will
            use FP16 precision for FP32/FP16 models, and BF16 precision for BF16.
        seed: Random seed for consistent reproducibility.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        download_dir: Optional[str],
        use_np_weights: bool,
        use_dummy_weights: bool,
        dtype: str,
        seed: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.download_dir = download_dir
        self.use_np_weights = use_np_weights
        self.use_dummy_weights = use_dummy_weights
        self.seed = seed

        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(model)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self._verify_tokenizer_mode()

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be either 'auto' or 'slow'.")
        self.tokenizer_mode = tokenizer_mode

    def _verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size}).")

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_heads(self, parallel_config: "ParallelConfig") -> int:
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

class CacheConfig:
    """Configuration for the KV cache.
    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the Aphrodite execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GiB
        self._verify_args()

        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. You passed "
                f"{self.gpu_memory_utilization} instead.")

    def _verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        num_gpu_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpu_per_node

        msg = (
            f"{cpu_memory_usage / _GiB:.2f} GiB out of "
            f"the {total_cpu_memory / _GiB:.2f} GiB total CPU memory is "
            "allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warn("Possibly too large swap space. " + msg)


class ParallelConfig:
    """Configuration for the distributed inference.
    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be
            set to `True` if either pipeline_parallel_size or
            tensor_parallel_size is greater than 1.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    """TODO(alpin): Implement pipeline parallelism."""
    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")

class SchedulerConfig:
    """Scheduler Configuration:
    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
    """
    def __init__(
        self,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str,
) -> torch.dtype:
    """Note: getattr(config, "torch_dtype", torch.float32) is incorrect
    because config.torch_dtype can be None"""
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    dtype = dtype.lower()
    # Check to see if dtype is a valid dtype *or* if it's auto
    if dtype not in [*_STR_DTYPE_TO_TORCH_DTYPE.values(), "auto"]:
        raise ValueError(f"Unknown dtype: {dtype}")
    
    # Obtain torch_dtype
    if dtype == "auto":
        if config_dtype == torch.float32:
            # Cast to 16-bit precision, BF16 if available
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.warning(f"Casting {config_dtype} to {torch_dtype}. Not recommended.")
        else:
            torch_dtype = config_dtype
    else:
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    return torch_dtype
