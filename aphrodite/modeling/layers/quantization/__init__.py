from typing import Type

import torch

from aphrodite.common.utils import is_hip

if torch.cuda.is_available():
    from aphrodite.modeling.layers.quantization.squeezellm import SqueezeLLMConfig
    from aphrodite.modeling.layers.quantization.gptq import GPTQConfig
    from aphrodite.modeling.layers.quantization.base_config import (
        QuantizationConfig)

_QUANTIZATION_CONFIG_REGISTRY = {
    "squeezellm": SqueezeLLMConfig if torch.cuda.is_available() else None,
    "gptq": GPTQConfig if torch.cuda.is_available() else None,
}

if not is_hip():
    from aphrodite.modeling.layers.quantization.awq import AWQConfig
    _QUANTIZATION_CONFIG_REGISTRY["awq"] = AWQConfig


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY or _QUANTIZATION_CONFIG_REGISTRY[quantization] is None:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig" if torch.cuda.is_available() else None,
    "get_quantization_config",
]
