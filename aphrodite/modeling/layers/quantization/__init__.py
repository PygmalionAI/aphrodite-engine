from typing import Type

from aphrodite.modeling.layers.quantization.awq import AWQConfig
from aphrodite.modeling.layers.quantization.gptq import GPTQConfig
from aphrodite.modeling.layers.quantization.base_config import QuantizationConfig

_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]

# class Linear:

#     @classmethod
#     def linear(cls, *args, **kwargs) -> nn.Module:
#         quant_config = kwargs.get("quant_config", None)
#         if quant_config is None:
#             kwargs.pop("quant_config", None)
#             return nn.Linear(*args, **kwargs)

#         name = quant_config.get_name()
#         if name not in _QUANTIZED_LINEAR_REGISTRY or _QUANTIZED_LINEAR_REGISTRY[
#                 name][2] is None:
#             raise ValueError(f"No quantized linear is found for {name}")

#         quant_linear_cls = _QUANTIZED_LINEAR_REGISTRY[name][2]
#         return quant_linear_cls(*args, **kwargs)


# class ParallelLinear:

#     @classmethod
#     def column(cls, *args, **kwargs) -> ColumnParallelLinear:
#         quant_config = kwargs.get("quant_config", None)
#         if quant_config is None:
#             return ColumnParallelLinear(*args, **kwargs)

#         name = quant_config.get_name()
#         if name not in _QUANTIZED_LINEAR_REGISTRY:
#             raise ValueError(f"No quantized linear is found for {name}")

#         quant_linear_cls = _QUANTIZED_LINEAR_REGISTRY[name][0]
#         return quant_linear_cls(*args, **kwargs)

#     @classmethod
#     def row(cls, *args, **kwargs) -> RowParallelLinear:
#         quant_config = kwargs.get("quant_config", None)
#         if quant_config is None:
#             return RowParallelLinear(*args, **kwargs)

#         name = quant_config.get_name()
#         if name not in _QUANTIZED_LINEAR_REGISTRY:
#             raise ValueError(f"No quantized linear is found for {name}")

#         quant_linear_cls = _QUANTIZED_LINEAR_REGISTRY[name][1]
#         return quant_linear_cls(*args, **kwargs)
