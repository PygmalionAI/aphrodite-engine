from typing import Type

from loguru import logger

from aphrodite.quantization.aqlm import AQLMConfig
from aphrodite.quantization.awq import AWQConfig
from aphrodite.quantization.base_config import QuantizationConfig
from aphrodite.quantization.bitsandbytes import BitsandBytesConfig
from aphrodite.quantization.deepspeedfp import DeepSpeedFPConfig
from aphrodite.quantization.eetq import EETQConfig
from aphrodite.quantization.exl2 import Exl2Config
from aphrodite.quantization.fp8 import Fp8Config
from aphrodite.quantization.gguf import GGUFConfig
from aphrodite.quantization.gptq import GPTQConfig
from aphrodite.quantization.gptq_marlin import GPTQMarlinConfig
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

QUANTIZATION_METHODS = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "bnb": BitsandBytesConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "eetq": EETQConfig,
    "exl2": Exl2Config,
    "fp8": Fp8Config,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "quip": QuipConfig,
    "squeezellm": SqueezeLLMConfig,
    "marlin": MarlinConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
