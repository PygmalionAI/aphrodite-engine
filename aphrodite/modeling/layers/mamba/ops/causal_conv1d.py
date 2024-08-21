# Copyright (c) 2024, Tri Dao.

from typing import Optional

import torch
from causal_conv1d_cuda import causal_conv1d_fwd, causal_conv1d_update


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out=None,
    activation: str = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None
    if seq_idx is not None:
        assert (initial_states is
                None), "initial_states must be None if seq_idx is not None"
        assert (not return_final_states
                ), "If seq_idx is not None, we don't return final_states_out"
    seq_idx = seq_idx.contiguous() if seq_idx is not None else None
    if initial_states is not None and (initial_states.stride(2) != 1
                                       and initial_states.stride(1) != 1):
        initial_states = initial_states.contiguous()
    if return_final_states:
        assert (
            x.stride(1) == 1
        ), "Only channel-last layout support returning final_states_out"
        if final_states_out is not None:
            assert (final_states_out.stride(2) == 1
                    or final_states_out.stride(1) == 1)
        else:
            batch, dim, seqlen = x.shape
            width = weight.shape[1]
            final_states_out = torch.empty(batch,
                                           width - 1,
                                           dim,
                                           device=x.device,
                                           dtype=x.dtype).transpose(1, 2)
    else:
        final_states_out = None

    out = causal_conv1d_fwd(x, weight, bias, seq_idx, initial_states,
                            final_states_out, activation in ["silu", "swish"])
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_up(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)
    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation = activation in ["silu", "swish"]
    return causal_conv1d_update(x, conv_state, weight, bias, activation)
