from typing import Type

from aphrodite.modeling.quantization_utils.awq import AWQConfig
from aphrodite.modeling.quantization_utils.gptq import GPTQConfig
from aphrodite.modeling.quantization_utils.base import QuantizationConfig

_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
}


def get_quant_class(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quant_class",
]
