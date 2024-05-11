from typing import Any, Dict, List, Optional
from contextlib import suppress

import torch

from aphrodite.modeling.layers.linear import (LinearMethodBase,
                                              set_weight_attrs)
from aphrodite.quantization.base_config import (QuantizationConfig)

HAS_QUANTS = False
with suppress(ImportError):
    from aphrodite._quant_C import quant_ops as ops
    HAS_QUANTS = True


def make_group_map(q_groups, num_qrows):
    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2

    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]
    return torch.tensor(group_map, dtype=torch.short, device=q_groups.device)


class Exl2Config(QuantizationConfig):
    """Config class for Exl2."""

    def __repr__(self) -> str:
        return "Exl2Config()"

    @classmethod
    def get_name(cls) -> str:
        return "exl2"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Exl2Config":
        return cls()

    def get_linear_method(self) -> "Exl2LinearMethod":
        return Exl2LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return False

    def quant_vocab(self) -> List[bool]:
        return [False, True]

    def support_fused_moe(self) -> bool:
        return False

    def rope_style(self) -> Optional[bool]:
        return None


class Exl2LinearMethod(LinearMethodBase):
    """Linear method for Exl2.

    Args:
        quant_config: The Exl2 quantization config.
    """

    def __init__(self, quant_config: Exl2Config):
        if not HAS_QUANTS:
            raise ImportError("Could not find the quantization kernels.")
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        # The shape of weight is unknown until load state dict
        # q_groups, q_invperm, q_scale, q_scale_max, q_weight, q_groups
        state_dict = {"exllama_state": 0}
        qweight = torch.nn.parameter.UninitializedParameter(
            requires_grad=False)
        set_weight_attrs(qweight, {"output_dim": 1, "ignore_warning": True})
        state_dict["q_weight"] = qweight
        qscale = torch.nn.parameter.UninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qscale, {
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": 8,
                "ignore_warning": True
            })
        state_dict["q_scale"] = qscale
        for name in ["q_groups", "q_invperm", "q_scale_max"]:
            fake_weight = torch.nn.parameter.UninitializedParameter(
                requires_grad=False)
            set_weight_attrs(fake_weight, {"ignore_warning": True})
            state_dict[name] = fake_weight
        return state_dict

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (weights["q_weight"].shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        if weights["exllama_state"] == 0:
            weights["q_scale_max"] /= 256
            weights["q_invperm"] = weights["q_invperm"].short()
            if "q_perm" not in weights:
                weights["q_perm"] = torch.argsort(weights["q_invperm"]).to(
                    torch.short)
            if "q_group_map" not in weights:
                weights["q_group_map"] = make_group_map(
                    weights["q_groups"], weights["q_weight"].shape[0])
            weights["q_matrix"] = ops.exl2_make_q_matrix(
                weights["q_weight"],
                weights["q_perm"],
                weights["q_invperm"],
                weights["q_scale"],
                weights["q_scale_max"],
                weights["q_groups"],
                weights["q_group_map"],
            )
            weights["exllama_state"] = 1

        output = ops.exl2_gemm(reshaped_x, weights["q_matrix"])

        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

    def apply_moe_weights(self, w1: Dict[str,
                                         torch.Tensor], w2: Dict[str,
                                                                 torch.Tensor],
                          x: torch.Tensor, gating_output: torch.Tensor,
                          topk: int, renormalize: bool) -> torch.Tensor:
        raise NotImplementedError
