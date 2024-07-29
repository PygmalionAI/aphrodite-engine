from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.quantization.base_config import QuantizationConfig


class TorchAOFPConfig(QuantizationConfig):
    """Config for TorchAO FP quantizer. It supports fp5, fp6 and fp7.
    
    Args: 
        weight_bits: the target quantization bits, 5, 6 or 7.
    """

    def __init__(
        self,
        weight_bits: int = 6,
        exp_bits: int = 2,
    ) -> None:
        self.weight_bits = weight_bits
        self.exponent_bits = exp_bits

        self.mantissa_bits = weight_bits - self.exponent_bits - 1

        self.valid_types = [torch.float16]

        if self.weight_bits not in (5, 6, 7):
            raise ValueError(
                "Currently, only 5-bit, 6-bit, and 7-bit weight"
                " quantization are "
                f"supported for TorchAO FP quantizaiton, but got "
                f"{self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"TorchAOFPConfig(weight_bits={self.weight_bits}), "
                f"exponent_bits={self.exponent_bits}")

    @classmethod
    def get_name(cls) -> str:
        return "TorchAOFP"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TorchAOFPConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        exp_bits = cls.get_from_keys(config, ["exp_bits"])
        return cls(weight_bits=weight_bits, exp_bits=exp_bits)

    def get_linear_method(self) -> "TorchAOFPLinearMethod":
        return TorchAOFPLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    def get_quant_method(
            self,
            layer: torch.nn.Module) -> Optional["TorchAOFPLinearMethod"]:
        if isinstance(layer, LinearBase):
            return TorchAOFPLinearMethod(self)
        return None


class TorchAOFPLinearMethod(LinearMethodBase):
    """Linear method for TorchAOFP quantizer.
    Args:
        quant_config: the TorchAOFP quantization config.
    """

    def __init__(self, quant_config: TorchAOFPConfig):
        self.quant_config = quant_config
        self.weight = None

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       weight_loader=None,
                       **extra_weight_attrs):
        del output_size
        del input_size
        output_size_per_partition = sum(output_partition_sizes)
        weight = TorchAOFPParameter(
            torch.Size((output_size_per_partition, input_size_per_partition)),
            params_dtype=params_dtype,
            quant_config=self.quant_config,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })
        layer.register_parameter("weight", weight)

        def quant_weight_loader(param, loaded_weight, *args, **kwargs):
            # Calls the original weight loader (if any), quantizes the result,
            # and then loads the quantized parameter.
            if weight_loader is not None:
                orig_param_data = param.data
                param.data = param.ao_dequantize()
                weight_loader(param, loaded_weight, *args, **kwargs)
                param.data, loaded_weight = orig_param_data, param.data
            param.ao_quantize_(loaded_weight.cuda())

        extra_weight_attrs["weight_loader"] = quant_weight_loader
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        weights = weight.data
        scales = weight.scales
        if bias is None:
            return weight.kernel(self.quant_config.exponent_bits,
                self.quant_config.mantissa_bits, x, weights, scales)
        else:
            return weight.kernel(self.quant_config.exponent_bits,
                self.quant_config.mantissa_bits, x, weights, scales) + bias

class TorchAOFPParameter(nn.Parameter):
    """
    TorchaAOFP quantized parameter class that implements fp5/fp6/fp7
    quantization. Weights are stored in quantized form on
    GPUs, and can be directly applied to float16 activations.
    """

    def __new__(cls, orig_shape: torch.Size, params_dtype: torch.dtype,
                quant_config: TorchAOFPConfig):
        try:
            from aphrodite.quantization.quant_llm.utils import (
                from_scaled_tc_fpx, to_scaled_tc_fpx)
            from aphrodite.quantization.quant_llm.utils.utils import \
                quant_llm_linear
        except ImportError as err:
            raise ImportError("Please install torchao via "
                              "`pip install torchao` to use "
                              "torchao quantizer.") from err

        data = torch.empty(torch.Size((orig_shape[0],
                            orig_shape[1] * quant_config.weight_bits // 8)),
                                   dtype=torch.uint8)


        self = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        self.scales = torch.empty(orig_shape[0],
                                  dtype=torch.float16)
        self.quant_config = quant_config
        self.orig_shape = orig_shape
        self.fp_quantizer = to_scaled_tc_fpx
        self.fp_dequantizer = from_scaled_tc_fpx
        self.kernel = quant_llm_linear
        return self

    def ao_quantize_(self, tensor: torch.Tensor):
        assert tensor.device.type == "cuda" and tensor.dtype != torch.int8
        data, scales = self.fp_quantizer(tensor.data, self.quant_config.exponent_bits,
                                                   self.quant_config.mantissa_bits)
        self.data.copy_(data)
        self.scales.copy_(scales)

    def ao_dequantize(self, output_dtype=None):
        output_dtype = output_dtype or torch.get_default_dtype()
        return self.fp_dequantizer(self.data, self.quant_config.exponent_bits, 
                        self.quant_config.mantissa_bits, self.scales).to(output_dtype)

