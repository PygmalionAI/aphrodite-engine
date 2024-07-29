# ruff: noqa
from functools import reduce
from typing import Tuple

import torch
from torch import Tensor

from aphrodite.quantization.quant_llm.utils.utils import (_f32_to_fpx_unpacked,
                                                          _fpx_unpacked_to_f32,
                                                          _n_ones)

_ONES_TABLE = [_n_ones(i) for i in range(8)]


def _pack(x: Tensor, n_bits: int) -> Tensor:
    return reduce(torch.bitwise_or, [x[..., i::(8 // n_bits)] << (8 - (i + 1) * n_bits) for i in range(8 // n_bits)])


def _unpack(x: Tensor, n_bits: int) -> Tensor:
    return torch.stack([(x >> (8 - (i + 1) * n_bits)) & ((1 << n_bits) - 1) for i in range(8 // n_bits)], dim=-1).flatten(-2)


# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h#L87-L116
def _bit_interleave(x: Tensor, n_bits: int, undo: bool = False) -> Tensor:
    # the original code unpacks/packs the values from/to uint32 while we unpack/pack the values from/to uint8
    # thus, we need to reverse byte order within a uint32 word.
    x = x.reshape(-1, 4).flip(1)

    x = _unpack(x, n_bits)
    x = x.view(-1, 4 * (8 // n_bits))

    if not undo:
        bit_order = {
            1: [1, 5, 9, 13, 17, 21, 25, 29, 3, 7, 11, 15, 19, 23, 27, 31,
                0, 4, 8, 12, 16, 20, 24, 28, 2, 6, 10, 14, 18, 22, 26, 30],
            2: [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14],
            4: [1, 5, 3, 7, 0, 4, 2, 6],
        }[n_bits]

    else:
        # this is inverse of the above, obtained by running
        # [v.index(i) for i in range(len(v))]
        bit_order = {
            1: [16, 0, 24, 8, 17, 1, 25, 9, 18, 2, 26, 10, 19, 3, 27, 11,
                20, 4, 28, 12, 21, 5, 29, 13, 22, 6, 30, 14, 23, 7, 31, 15],
            2: [8, 0, 12, 4, 9, 1, 13, 5, 10, 2, 14, 6, 11, 3, 15, 7],
            4: [4, 0, 6, 2, 5, 1, 7, 3],
        }[n_bits]

    x = x[:, bit_order]
    x = _pack(x, n_bits)

    # reverse byte order within a uint32 word again.
    x = x.reshape(-1, 4).flip(1)
    return x.flatten()


# this is a literal adaptation of FP6-LLM ahead-of-time bit-level pre-packing
# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h
def _pack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    assert tensor.ndim == 2, tensor.dtype == torch.uint8
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    # Pass 1 from original code
    tensor = tensor.view(M // 64, 4, 2, 8, N // 16, 2, 8)
    tensor = tensor.permute(0, 4, 1, 5, 2, 3, 6)
    tensor = tensor.reshape(-1, 32, 2)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor.flatten()

    used_bits = 0
    fragments = []

    for y in [1, 2, 4]:
        if nbits & y:
            mask = (1 << y) - 1
            tensor_ybit = (tensor >> (nbits - used_bits - y)) & mask
            tensor_ybit = _pack(tensor_ybit, y)

            tensor_ybit = tensor_ybit.view(32, -1, 4).permute(1, 0, 2).flip(2)
            tensor_ybit = _bit_interleave(tensor_ybit.flatten(), y)
            fragments.append(tensor_ybit)
            used_bits += y

    return torch.cat(fragments, dim=0).view(M, -1)


# more optimized version of _pack_tc_fpx() for FP6 by merging ops
def _pack_tc_fp6(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2, tensor.dtype == torch.uint8
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor = tensor.view(M // 64, 2, 2, 2, 8, N // 16, 2, 8)
    tensor = tensor.flip(3)

    tensor_2bit = (tensor >> 4) & 0b11
    tensor_2bit = tensor_2bit.permute(0, 5, 1, 4, 7, 3, 2, 6)
    tensor_2bit = _pack(tensor_2bit.flatten(), 2)

    tensor_4bit = tensor & 0b1111
    tensor_4bit = tensor_4bit.permute(0, 5, 1, 2, 4, 7, 3, 6)
    tensor_4bit = _pack(tensor_4bit.flatten(), 4)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0).view(M, -1)


# currently only optimize for TC-FP6 packing
def pack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    if nbits == 6:
        return _pack_tc_fp6(tensor)
    return _pack_tc_fpx(tensor, nbits)


def to_scaled_tc_fpx(tensor: Tensor, ebits: int, mbits: int) -> Tuple[Tensor, Tensor]:
    # _n_ones() is not compatible with torch.compile() due to << operator
    # https://github.com/pytorch/pytorch/issues/119152
    # exp_bias = _n_ones(ebits - 1)
    # max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2 ** mbits))

    # workaround: global lookup table
    exp_bias = _ONES_TABLE[ebits - 1]
    max_normal = 2 ** (_ONES_TABLE[ebits] - exp_bias) * (_ONES_TABLE[mbits + 1] / (2 ** mbits))

    tensor = tensor.float()
    scale = tensor.abs().amax(1).clamp(min=1e-12) / max_normal
    tensor_fpx = _f32_to_fpx_unpacked(tensor / scale.view(-1, 1), ebits, mbits)
    tensor_tc_fpx = pack_tc_fpx(tensor_fpx, 1 + ebits + mbits)
    return tensor_tc_fpx, scale.half()


# inverse of _pack_tc_fpx()
def _unpack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    size = tensor.numel()
    tensor = tensor.flatten()
    offset = 0
    used_bits = 0

    tensor_fpx = None

    for y in [1, 2, 4]:
        if nbits & y:
            size_ybit = size // nbits * y
            tensor_ybit = tensor[offset : offset + size_ybit]
            offset += size_ybit

            tensor_ybit = _bit_interleave(tensor_ybit, y, undo=True)            # undo Pass 3
            tensor_ybit = tensor_ybit.view(-1, 32, 4).flip(2).permute(1, 0, 2)  # undo Pass 2

            tensor_ybit = _unpack(tensor_ybit.flatten(), y)
            tensor_ybit = tensor_ybit << (nbits - used_bits - y)
            used_bits += y

            if tensor_fpx is None:
                tensor_fpx = tensor_ybit
            else:
                tensor_fpx |= tensor_ybit

    # undo Pass 1
    tensor_fpx = tensor_fpx.view(32, -1, 2).permute(1, 0, 2)
    tensor_fpx = tensor_fpx.reshape(M // 64, -1, 4, 2, 2, 8, 8)
    tensor_fpx = tensor_fpx.permute(0, 2, 4, 5, 1, 3, 6)
    tensor_fpx = tensor_fpx.reshape(M, -1)
    return tensor_fpx


# more optimized version of _unpack_tc_fpx() for FP6 by merging ops
# inverse of _unpack_tc_fp6()
def _unpack_tc_fp6(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    N = tensor.shape[1] // 3 * 4
    assert (M % 64 == 0) and (N % 64 == 0)
    size_2bit = M * N // 4
    size_4bit = M * N // 2
    tensor = tensor.view(-1)
    assert tensor.numel() == size_2bit + size_4bit

    tensor_2bit, tensor_4bit = tensor.split([size_2bit, size_4bit])

    tensor_2bit = _unpack(tensor_2bit, 2)
    tensor_2bit = tensor_2bit.view(M // 64, N // 16, 2, 8, 8, 2, 2, 2)
    tensor_2bit = tensor_2bit.permute(0, 2, 6, 5, 3, 1, 7, 4)

    tensor_4bit = _unpack(tensor_4bit, 4)
    tensor_4bit = tensor_4bit.view(M // 64, N // 16, 2, 2, 8, 8, 2, 2)
    tensor_4bit = tensor_4bit.permute(0, 2, 3, 6, 4, 1, 7, 5)

    tensor_fp6 = (tensor_2bit << 4) | tensor_4bit
    tensor_fp6 = tensor_fp6.flip(3).reshape(M, N)
    return tensor_fp6


def unpack_tc_fpx(tensor: Tensor, nbits: int) -> Tensor:
    if nbits == 6:
        return _unpack_tc_fp6(tensor)
    return _unpack_tc_fpx(tensor, nbits)


def from_scaled_tc_fpx(tensor: Tensor, ebits: int, mbits: int, scale=None) -> Tensor:
    fpx_unpacked = unpack_tc_fpx(tensor, 1 + ebits + mbits)
    tensor = _fpx_unpacked_to_f32(fpx_unpacked, ebits, mbits)
    if scale is not None:
        tensor = tensor * scale.float().view(-1, 1)
    return tensor
