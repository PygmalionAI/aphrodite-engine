from typing import Type

from aphrodite.quantization.aqlm import AQLMConfig
from aphrodite.quantization.awq import AWQConfig
from aphrodite.quantization.awq_marlin import AWQMarlinConfig
from aphrodite.quantization.base_config import QuantizationConfig
from aphrodite.quantization.bitsandbytes import BitsAndBytesConfig
from aphrodite.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig)
from aphrodite.quantization.deepspeedfp import DeepSpeedFPConfig
from aphrodite.quantization.eetq import EETQConfig
from aphrodite.quantization.fbgemm_fp8 import FBGEMMFp8Config
from aphrodite.quantization.fp8 import Fp8Config
from aphrodite.quantization.gguf import GGUFConfig
from aphrodite.quantization.gptq import GPTQConfig
from aphrodite.quantization.gptq_marlin import GPTQMarlinConfig
from aphrodite.quantization.gptq_marlin_24 import GPTQMarlin24Config
from aphrodite.quantization.marlin import MarlinConfig
from aphrodite.quantization.qqq import QQQConfig
from aphrodite.quantization.quip import QuipConfig
from aphrodite.quantization.squeezellm import SqueezeLLMConfig
from aphrodite.quantization.tpu_int8 import Int8TpuConfig

QUANTIZATION_METHODS = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "tpu_int8": Int8TpuConfig,
    "eetq": EETQConfig,
    "fp8": Fp8Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "gguf": GGUFConfig,
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin": MarlinConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "gptq_marlin": GPTQMarlinConfig,
    "awq_marlin": AWQMarlinConfig,
    "gptq": GPTQConfig,
    "quip": QuipConfig,
    "squeezellm": SqueezeLLMConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
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
