from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from aphrodite.modeling.layers.linear import LinearMethodBase


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    @abstractmethod
    def get_name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's "
                         "quantization config.")

    @classmethod
    def get_packed_tensors(cls) -> Dict[str, int]:
        """Returns a dictionary of packed tensor names and their pack dims."""
        raise NotImplementedError

    @classmethod
    def get_packed_dim(cls, tensor_name: str) -> Optional[int]:
        """Returns the pack dim of a tensor if it is packed.

        A tensor is considered packed if each element in the tensor is a
        packed representation of multiple elements in the original tensor.
        For example, an INT32 element in the tensor may represent 8 INT4
        elements in the original tensor.
        If the tensor is not packed, returns None.
        """
        packed_tensors = cls.get_packed_tensors()
        for packed_tensor_name, pack_dim in packed_tensors.items():
            if packed_tensor_name in tensor_name:
                return pack_dim
        return None

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def is_transposed(cls, tensor_name: str) -> bool:
        """Returns True if a tensor is transposed relative to nn.Linear.weight.
        """
        return any(tag in tensor_name
                   for tag in cls.get_transposed_tensor_names())

    def get_col_parallel_tensor_names(self) -> List[str]:
        raise NotImplementedError

    def get_row_parallel_tensor_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_linear_method(self) -> LinearMethodBase:
        """Get the linear method to use for the quantized linear layer."""
        raise NotImplementedError

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError
