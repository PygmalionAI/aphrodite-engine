from typing import Type

from loguru import logger

from aphrodite.quantization.aqlm import AQLMConfig
from aphrodite.quantization.awq import AWQConfig
from aphrodite.quantization.base_config import \
    QuantizationConfig
from aphrodite.quantization.bitsandbytes import \
    BitsandBytesConfig
from aphrodite.quantization.eetq import EETQConfig
from aphrodite.quantization.exl2 import Exl2Config
from aphrodite.quantization.gguf import GGUFConfig
from aphrodite.quantization.gptq import GPTQConfig
from aphrodite.quantization.marlin import MarlinConfig
from aphrodite.quantization.quip import QuipConfig
from aphrodite.quantization.squeezellm import SqueezeLLMConfig

try:
    from aphrodite._quant_C import quant_ops  # noqa: F401
except ImportError:
    logger.warning("The Quantization Kernels are not installed. "
                   "To use quantization with Aphrodite, make sure "
                   "you've exported the `APHRODITE_INSTALL_QUANT_KERNELS=1`"
                   "environment variable during the compilation process.")

_QUANTIZATION_CONFIG_REGISTRY = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "bnb": BitsandBytesConfig,
    "eetq": EETQConfig,
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
