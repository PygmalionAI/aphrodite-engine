from typing import Type

from aphrodite.modeling.layers.quantization.squeezellm import SqueezeLLMConfig
from aphrodite.modeling.layers.quantization.gptq import GPTQConfig
from aphrodite.modeling.layers.quantization.base_config import (
    QuantizationConfig)
from aphrodite.common.utils import is_hip

_QUANTIZATION_CONFIG_REGISTRY = {
    "squeezellm": SqueezeLLMConfig,
    "gptq": GPTQConfig,
}

if not is_hip():
    from aphrodite.modeling.layers.quantization.awq import AWQConfig
    _QUANTIZATION_CONFIG_REGISTRY["awq"] = AWQConfig


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]
