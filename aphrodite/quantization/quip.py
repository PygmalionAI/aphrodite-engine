from typing import Any, Dict, List, Optional
from contextlib import suppress

import torch
from torch.nn.parameter import Parameter

from aphrodite.modeling.layers.linear import (LinearMethodBase,
                                              set_weight_attrs)
from aphrodite.quantization.base_config import (QuantizationConfig)
from aphrodite.quantization.quip_utils import (
    get_packed_abs_grid,
    get_hadK,
    matmul_hadUt_cuda,
    matmul_hadU_cuda,
)

HAS_QUANTS = False
with suppress(ImportError):
    from aphrodite._quant_C import quant_ops as ops
    HAS_QUANTS = True


class QuipConfig(QuantizationConfig):
    """Config class for Quip.

    Reference: https://cornell-relaxml.github.io/quip-sharp/
    """

    def __init__(self, codebook: int, use_rand: bool) -> None:
        if not HAS_QUANTS:
            raise ImportError("Could not find the quantization kernels.")
        self.codebook = codebook
        self.use_rand = use_rand

        if self.codebook != "E8P12":
            raise ValueError("Currently, only E8P12 is supported for "
                             f"Quip, but got {self.codebook}.")

    def __repr__(self) -> str:
        return (f"QuipConfig(codebook={self.codebook}, "
                f"rescale_WH={self.rescale_WH})")

    @classmethod
    def get_name(cls) -> str:
        return "quip"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuipConfig":
        codebook = cls.get_from_keys(config, ["codebook"])
        use_rand = cls.get_from_keys(config, ["use_rand"])
        return cls(codebook, use_rand)

    def get_linear_method(self) -> "QuipLinearMethod":
        return QuipLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    def merge_weight(self) -> bool:
        return False

    def quant_vocab(self) -> List[bool]:
        return [False, False]

    def support_fused_moe(self) -> bool:
        return False

    def rope_style(self) -> Optional[bool]:
        return None


class QuipLinearMethod(LinearMethodBase):
    """Linear method for Quip.

    Args:
        quant_config: The Quip quantization config.
    """

    def __init__(self, quant_config: QuipConfig):
        self.quant_config = quant_config
        self.grid_packed_abs = get_packed_abs_grid().to(device="cuda")
        self.pack = 8
        self.idx_dtype = torch.int16

    def create_weights(
        self,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        output_size_per_partition = sum(output_partition_sizes)
        if (input_size != input_size_per_partition
                or output_size != output_size_per_partition):
            raise ValueError(
                "Currently Quip doesn't support tensor parallel yet")

        had_left, K_left, q_in_features = get_hadK(input_size,
                                                   self.quant_config.use_rand)
        had_right, K_right, q_out_features = get_hadK(
            output_size, self.quant_config.use_rand)
        weights = {
            "K_left": K_left,
            "K_right": K_right,
            "q_in_features": q_in_features,
            "q_out_features": q_out_features,
        }
        if had_left is not None:
            weights["had_left"] = Parameter(
                had_left.to(dtype=params_dtype, device="cuda"),
                requires_grad=False,
            )
            set_weight_attrs(weights["had_left"], {"ignore_warning": True})
        if had_right is not None:
            weights["had_right"] = Parameter(
                had_right.to(dtype=params_dtype, device="cuda"),
                requires_grad=False,
            )
            set_weight_attrs(weights["had_right"], {"ignore_warning": True})
        Qidxs = Parameter(
            torch.empty(q_out_features,
                        q_in_features // self.pack,
                        device="cuda",
                        dtype=self.idx_dtype),
            requires_grad=False,
        )
        set_weight_attrs(Qidxs, {"ignore_warning": True})
        Wscale = Parameter(
            torch.ones((), dtype=torch.float, device="cuda"),
            requires_grad=False,
        )
        set_weight_attrs(Wscale, {"ignore_warning": True})
        SU = Parameter(
            torch.ones(
                input_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(SU, {"ignore_warning": True})
        SV = Parameter(
            torch.ones(
                output_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(SV, {"ignore_warning": True})
        weights.update({
            "Qidxs": Qidxs,
            "Wscale": Wscale,
            "SU": SU,
            "SV": SV,
        })
        return weights

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First run
        if isinstance(weights["Wscale"], torch.Tensor):
            weights["Wscale"] = weights["Wscale"].item()
            if "SU" in weights and torch.all(weights["SU"] > 0):
                del weights["SU"]
            if "SV" in weights and torch.all(weights["SV"] > 0):
                del weights["SV"]

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_dim = weights["Qidxs"].shape[0]

        if "SU" in weights:
            reshaped_x = reshaped_x * weights["SU"]
        reshaped_x = matmul_hadUt_cuda(reshaped_x,
                                       weights.get("had_left",
                                                   None), weights["K_left"],
                                       weights["q_in_features"],
                                       weights["Wscale"])

        m, n = weights["Qidxs"].shape
        if reshaped_x.size(0) < 32:
            out = ops.quip_gemv(reshaped_x, weights["Qidxs"],
                                self.grid_packed_abs)
        else:
            W_decompressed = torch.empty(m,
                                         n * 8,
                                         dtype=torch.float16,
                                         device=x.device)
            ops.quip_decompress(weights["Qidxs"], self.grid_packed_abs,
                                W_decompressed)
            out = reshaped_x @ W_decompressed.T

        out = matmul_hadU_cuda(out, weights.get("had_right",
                                                None), weights["K_right"],
                               weights["q_out_features"])[..., :out_dim]
        if "SV" in weights:
            out = out * weights["SV"]
        out = out.view(*x.shape[:-1], out.shape[-1])
        out = out + bias if bias is not None else out
        return out

    def apply_moe_weights(self, w1: Dict[str,
                                         torch.Tensor], w2: Dict[str,
                                                                 torch.Tensor],
                          x: torch.Tensor, gating_output: torch.Tensor,
                          topk: int, renormalize: bool) -> torch.Tensor:
        raise NotImplementedError
