from typing import Optional
import torch
import torch.nn as nn

from aphrodite import activation_ops
from aphrodite.modeling.layers.quantization import QuantizationConfig


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    TODO(alpin): Add more activation functions.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out


class NewGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        activation_ops.gelu_new(out, x)
        return out


class FastGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        activation_ops.gelu_fast(out, x)
        return out


class ScaledActivation(nn.Module):

    def __init__(
        self,
        act_module: nn.Module,
        hidden_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.act_module = act_module
        self.scales = nn.Parameter(
            torch.empty(hidden_size, dtype=params_dtype, device="cuda"))

    def forward(self, x: torch.Tensor):
        return self.act(x) / self.scales


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_new": NewGELU(),
    "gelu_fast": FastGELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
}


def get_act_fn(
    act_fn: str,
    quant_config: Optional[QuantizationConfig] = None,
    intermediate_size: Optional[int] = None,
) -> nn.Module:
    """Get an activation function by name."""
    # pylint: disable=used-before-assignment
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn!r} is currently not supported.")
    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if quant_config is not None:
        if act_fn_name in quant_config.get_scaled_act_names():
            if intermediate_size is None:
                raise ValueError(
                    "intermediate_size must be provided when using "
                    f"{act_fn_name} with quantization for scaled "
                    "activation functions.")
            return ScaledActivation(
                act_fn,
                intermediate_size,
                params_dtype=torch.get_default_dtype(),
            )
    return act_fn
