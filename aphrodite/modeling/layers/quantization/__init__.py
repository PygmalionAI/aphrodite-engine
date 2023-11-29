from typing import Type

from aphrodite.modeling.layers.quantization.awq import AWQConfig
from aphrodite.modeling.layers.quantization.base_config import QuantizationConfig

_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": AWQConfig,

}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]

