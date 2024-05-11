from typing import Any, Dict, List, Optional
from contextlib import suppress

import torch
from torch.nn.parameter import Parameter

from aphrodite.modeling.layers.linear import (LinearMethodBase,
                                              set_weight_attrs)
from aphrodite.quantization.base_config import (QuantizationConfig)

HAS_QUANTS = False
with suppress(ImportError):
    from aphrodite._quant_C import quant_ops as ops
    HAS_QUANTS = True

GGML_QUANT_SIZES = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 2 + 16),  # Q4_0
    3: (32, 2 + 2 + 16),  # Q4_1
    6: (32, 2 + 4 + 16),  # Q5_0
    7: (32, 2 + 2 + 4 + 16),  # Q5_1
    8: (32, 2 + 32),  # Q8_0
    9: (32, 4 + 4 + 32),  # Q8_1
    10: (256, 2 + 2 + 256 // 16 + 256 // 4),  # Q2_K
    11: (256, 2 + 256 // 4 + 256 // 8 + 12),  # Q3_K
    12: (256, 2 + 2 + 256 // 2 + 12),  # Q4_K
    13: (256, 2 + 2 + 256 // 2 + 256 // 8 + 12),  # Q5_K
    14: (256, 2 + 256 // 2 + 256 // 4 + 256 // 16),  # Q6_K
    15: (256, 4 + 256 + 256 // 8),  # Q8_K
    16: (256, 2 + 256 // 4),  # IQ2_XXS
    17: (256, 2 + 256 // 4 + 256 // 32),  # IQ2_XS
    18: (256, 2 + 3 * 256 // 8),  # IQ3_XXS
    19: (256, 2 + 256 // 8 + 256 // 16),  # IQ1_S
    20: (32, 2 + 32 // 2),  # IQ4_NL
    21: (256, 2 + 256 // 4 + 256 // 32 + 256 // 8 + 256 // 64),  # IQ3_S
    22: (256, 2 + 256 // 4 + 256 // 32 + 256 // 32),  # IQ2_S
    23: (256, 2 + 2 + 256 // 64 + 256 // 2),  # IQ4_XS
}


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF"""

    def __repr__(self) -> str:
        return ("GGUFConfig()")

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        return 61

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_linear_method(self) -> "GGUFLinearMethod":
        return GGUFLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return False

    def rope_style(self) -> Optional[bool]:
        return False

    def quant_vocab(self) -> List[bool]:
        return [True, True]

    def support_fused_moe(self) -> bool:
        return False


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        if not HAS_QUANTS:
            raise ImportError("Could not find the quantization kernels.")
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        # The type of weight is unknown until load state dict
        weight = torch.nn.parameter.UninitializedParameter(requires_grad=False)
        # No need for pack_factor because we don't fuse qkv layers anyway.
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })
        weight_type = Parameter(
            torch.tensor((1), dtype=torch.int, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(weight_type, {"ignore_warning": True})
        return {"weight": weight, "weight_type": weight_type}

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(weights["weight_type"], torch.Tensor):
            weights["weight_type"] = int(weights["weight_type"])
            # Check tensor parallel shape here on first pass
            block_size = GGML_QUANT_SIZES[weights["weight_type"]][1]
            if weights["weight"].shape[1] % block_size != 0:
                raise ValueError("Size is not aligned with the quantized "
                                 "weight shape.")

        weight = weights["weight"]
        weight_type = weights["weight_type"]
        infeatures = x.shape[-1]
        outfeatures = weight.shape[0]
        out_shape = x.shape[:-1] + (weight.shape[0], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        xshape = x.view(-1, x.shape[-1])
        if xshape.shape[0] == 1:
            out = ops.ggml_mul_mat_vec_a8(weight, reshaped_x, weight_type,
                                          outfeatures)
        elif xshape.shape[0] < 8 and weight_type < 16:
            out = ops.ggml_mul_mat_a8(weight, reshaped_x, weight_type,
                                      outfeatures)
        else:
            weight = ops.ggml_dequantize(weight, weight_type, outfeatures,
                                         infeatures)
            out = reshaped_x @ weight.T

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)

    def apply_embedding(self, weights: Dict[str, torch.Tensor],
                        x: torch.Tensor) -> torch.Tensor:
        if isinstance(weights["weight_type"], torch.Tensor):
            weights["weight_type"] = int(weights["weight_type"])
        weight = weights["weight"]
        weight_type = weights["weight_type"]
        dim, block_size = GGML_QUANT_SIZES[weights["weight_type"]]
        vocab_size = weight.shape[0]
        hidden_size = weight.shape[1] // block_size * dim
        if weight_type < 2:
            return torch.embedding(weight.view(vocab_size, -1), x)
        x_flat = x.flatten()
        quant = torch.index_select(weight.view(vocab_size, -1),
                                   dim=0,
                                   index=x_flat)
        dequant = ops.ggml_dequantize(quant, weight_type, hidden_size,
                                      x_flat.shape[0])
        return dequant.view(*x.shape, hidden_size)

    def apply_moe_weights(self, w1: Dict[str,
                                         torch.Tensor], w2: Dict[str,
                                                                 torch.Tensor],
                          x: torch.Tensor, gating_output: torch.Tensor,
                          topk: int, renormalize: bool) -> torch.Tensor:
        raise NotImplementedError
