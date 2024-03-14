from typing import Type

from aphrodite.modeling.layers.quantization.base_config import QuantizationConfig
from aphrodite.modeling.layers.quantization.aqlm import AQLMConfig
from aphrodite.modeling.layers.quantization.awq import AWQConfig
from aphrodite.modeling.layers.quantization.bitsandbytes import BitsandBytesConfig
from aphrodite.modeling.layers.quantization.exl2 import Exl2Config
from aphrodite.modeling.layers.quantization.gguf import GGUFConfig
from aphrodite.modeling.layers.quantization.gptq import GPTQConfig
from aphrodite.modeling.layers.quantization.quip import QuipConfig
from aphrodite.modeling.layers.quantization.squeezellm import SqueezeLLMConfig
from aphrodite.modeling.layers.quantization.marlin import MarlinConfig

_QUANTIZATION_CONFIG_REGISTRY = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "bnb": BitsandBytesConfig,
    "exl2": Exl2Config,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "quip": QuipConfig,
    "squeezellm": SqueezeLLMConfig,
    "marlin": MarlinConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]
