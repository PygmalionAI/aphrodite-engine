import enum
from enum import Enum
from typing import Any, Dict, List, Optional
from fractions import Fraction

import numpy as np
import torch
from torch.nn.parameter import Parameter

from aphrodite._C import ops as quantization_ops
from aphrodite.common.utils import is_hip
from aphrodite.modeling.layers.linear import (LinearMethodBase,
                                              set_weight_attrs)
from aphrodite.modeling.layers.quantization.base_config import (
    QuantizationConfig)

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

def pemute_weight(qweight, scale, group_size, g_idx=None):
    # unpack and permute qweight
    w = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 8, -1),
        torch.tensor(list(range(0, 32, 4)), dtype=torch.int32, device=qweight.device
                     ).unsqueeze(0).unsqueeze(-1),
        ).bitwise_and(15)
    w = w.reshape(-1, w.shape[2]).contiguous()
    if g_idx is not None:
        w = w[g_idx, :]
    tile = 16
    w = w.reshape((w.shape[0] // tile, tile, w.shape[1] // tile, tile))
    w = w.permute((0, 2, 1, 3)).reshape(w.shape[0], -1)
    res = w.reshape((-1, _perm.numel()))[:, _perm].reshape(w.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i
    q = torch.from_numpy(q.astype(np.int32)).to(w.device)
    # permute scale
    dim = scale.shape[1]
    if group_size == -1:
        scale= scale.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    else:
        scale = scale.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    scale = scale.reshape((-1, dim)).contiguous()
    return q, scale

class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is supported for "
                f"GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"sym={self.sym}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        sym = cls.get_from_keys(config, ["sym"])
        return cls(weight_bits, group_size, desc_act, sym)

    def get_linear_method(self) -> "GPTQLinearMethod":
        return GPTQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []
    
    def merge_weight(self) -> bool:
        return True

    def rope_style(self) -> Optional[bool]:
        return None


class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()
    MARLIN_UNINITIALIZED = enum.auto()
    MARLIN_READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.workspace = torch.zeros((512,), dtype=torch.int, device="cuda")

    def fit_marlin(self, output_size):
        return self.quant_config.group_size in (-1, 128) and (
            self.quant_config.weight_bits == 4) and (
            self.quant_config.sym) and (
            not self.quant_config.desc_act) and (
            output_size % 256 == 0) and not is_hip()

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        del output_size  # Unused.
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if (output_size_per_partition % self.quant_config.pack_factor.numerator
                != 0):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        # For act-order models, we cannot use Exllama for row parallel layer
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0
                if self.fit_marlin(output_size_per_partition):
                    exllama_state = ExllamaState.MARLIN_UNINITIALIZED
        elif self.fit_marlin(output_size_per_partition):
            exllama_state = ExllamaState.MARLIN_UNINITIALIZED

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })
        g_idx = Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})
        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": scale_and_zero_input_dim,
            "output_dim": 1,
        })
        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "qzeros": qzeros,
            "scales": scales,
            "exllama_state": exllama_state,
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (weights["scales"].shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if weights["exllama_state"] == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                weights["g_idx"] = torch.argsort(weights["g_idx"]).to(
                    torch.int)
            else:
                weights["g_idx"] = torch.empty((1, 1), device="meta")
            weights["exllama_state"] = ExllamaState.READY
            quantization_ops.gptq_shuffle(weights["qweight"], weights["g_idx"],
                                          self.quant_config.weight_bits)
        elif weights["exllama_state"] == ExllamaState.MARLIN_UNINITIALIZED:
            if self.quant_config.desc_act:
                weights["g_idx"] = torch.argsort(weights["g_idx"]).to(
                    torch.int)
            else:
                weights["g_idx"] = None
            weights["qweight"], weights["scales"] = pemute_weight(weights["qweight"],
                                                                  weights["scales"],
                                                                  self.quant_config.group_size,
                                                                  weights["g_idx"])
            weights["exllama_state"] = ExllamaState.MARLIN_READY

        if weights["exllama_state"] == ExllamaState.MARLIN_READY:
            output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            # reorder input for act-order model
            if weights["g_idx"] is not None:
                reshaped_x = reshaped_x[:, weights["g_idx"]]
            quantization_ops.marlin_gemm(reshaped_x, weights["qweight"],
                                         output.view(-1, output.shape[-1]),
                                         weights["scales"], self.workspace)
        else:
            output = quantization_ops.gptq_gemm(reshaped_x, weights["qweight"],
                                                weights["qzeros"], weights["scales"],
                                                weights["g_idx"],
                                                weights["exllama_state"] == ExllamaState.READY,
                                                self.quant_config.weight_bits)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
