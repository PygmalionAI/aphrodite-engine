from typing import Optional, Tuple, Type

import torch

from aphrodite._quant_C import quant_ops


# cutlass
def cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
                         scale_a: torch.Tensor, scale_b: torch.Tensor,
                         out_dtype: Type[torch.dtype]) -> torch.Tensor:
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    quant_ops.cutlass_scaled_mm_dq(out, a, b, scale_a, scale_b)

    return out


# int8
def scaled_int8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale.
    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
    Returns:
      Tuple[Torch.Tensor, Torch.Tensor] : Output int8 tensor and scales.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        quant_ops.static_scaled_int8_quant(output, input, scale)
        return output, scale

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    quant_ops.dynamic_scaled_int8_quant(output, input, input_scales)
    return output, input_scales
