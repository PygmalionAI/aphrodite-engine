# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 candidate-consumer scoring for SM120's 128-row planar KV cache."""

import torch

from aphrodite.model_executor.kernels.attention.dsa.sm120_indexer import _fp8_paged_cache_views
from aphrodite.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    _sparse_topk_remap_kernel,
    candidate_blocks_to_sparse_indices,
)
from aphrodite.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
from aphrodite.triton_utils import tl, triton
from aphrodite.v1.worker.workspace import current_workspace_manager


@triton.jit
def _sm120_candidate_logits(
    Q,
    K,
    Scale,
    W,
    Lens,
    Table,
    Sparse,
    End,
    Out,
    stride_qb: tl.int64,
    stride_qn: tl.int64,
    stride_qh: tl.int64,
    stride_qd: tl.int64,
    stride_kp: tl.int64,
    stride_kt: tl.int64,
    stride_sp: tl.int64,
    stride_st: tl.int64,
    stride_w: tl.int64,
    stride_tb: tl.int64,
    stride_tc: tl.int64,
    stride_si: tl.int64,
    stride_out: tl.int64,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NEXT_N
    qi = row % NEXT_N
    heads = tl.arange(0, HEADS)[:, None]
    dims = tl.arange(0, DIM)
    length = tl.minimum(tl.maximum(tl.load(Lens + row), 0), MAX_LEN)
    end = tl.minimum(tl.maximum(tl.load(End + row), 0), WIDTH)
    q = tl.load(Q + batch * stride_qb + qi * stride_qn + heads * stride_qh + dims[None, :] * stride_qd)
    weights = tl.load(W + row * stride_w + heads).to(tl.float32)
    # Static launch, live-prefix work. Zero rows never dereference a page.
    for tile in range(tl.program_id(1), tl.cdiv(end, BLOCK), tl.num_programs(1)):
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        sb = tl.load(Sparse + row * stride_si + cols // 8, cols < WIDTH, other=0)
        pos = sb * 8 + cols % 8
        logical_page = pos // PAGE
        valid = (cols < end) & (cols < WIDTH) & (pos >= 0) & (pos < length) & (logical_page < TABLE_COLS)
        physical = tl.load(Table + batch * stride_tb + logical_page * stride_tc, valid, other=0)
        valid &= (physical >= 0) & (physical < PAGES)
        slot = pos % PAGE
        k = tl.load(
            K + physical[None, :] * stride_kp + slot[None, :] * stride_kt + dims[:, None], valid[None, :], other=0.0
        )
        scale = tl.load(Scale + physical * stride_sp + slot * stride_st, valid, other=0.0).to(tl.float32)
        dot = tl.dot(q, k, input_precision="ieee")
        scores = tl.maximum(dot.to(tl.float32) * scale[None, :], 0.0)
        scores = tl.sum(scores * weights, axis=0)
        tl.store(Out + row * stride_out + cols, tl.where(valid, scores, -float("inf")), cols < WIDTH)


def _scratch_specs(rows: int) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    return (
        ((rows, 2048), torch.int32),
        ((rows,), torch.int32),
        ((rows, 16384), torch.float32),
        ((rows, 512), torch.int32),
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )


def _row_starts(capacity: int) -> torch.Tensor:
    return current_workspace_manager().get_persistent(
        ("sm120_fp8_candidate_starts", capacity),
        (capacity,),
        torch.int32,
        zero_init=True,
    )


def reserve_sm120_candidate_topk(capacity: int) -> None:
    """Reserve lane/ubatch-owned scratch before CUDA graph capture."""
    _row_starts(capacity)
    current_workspace_manager().get_simultaneous(*_scratch_specs(capacity))


def sm120_fp8_candidate_topk(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    tables: torch.Tensor,
    candidates: torch.Tensor,
    max_model_len: int,
    output: torch.Tensor,
) -> None:
    """Select top-512 from 2048 unique 8-token candidate blocks per query.

    Callers admit contiguous FP8 [batch, next_n, 32, 128] queries and FP32
    weights with a contiguous head dimension. Only the live score prefix is
    initialized; persistent_topk consumes the same per-row candidate counts.
    """
    batch, next_n, heads, dim = q.shape
    rows = batch * next_n
    sparse, ends, logits, columns, topk_workspace = current_workspace_manager().get_simultaneous(*_scratch_specs(rows))
    starts = _row_starts(candidates.shape[0])[:rows]
    candidate_blocks_to_sparse_indices(candidates[:rows], starts, lengths.reshape(-1), 8, 8, out=(sparse, ends))
    keys, scales = _fp8_paged_cache_views(cache, dim)
    width = logits.shape[1]
    _sm120_candidate_logits[(rows, 128)](
        q,
        keys,
        scales,
        weights,
        lengths,
        tables,
        sparse,
        ends,
        logits,
        *q.stride(),
        keys.stride(0),
        keys.stride(1),
        *scales.stride(),
        weights.stride(0),
        *tables.stride(),
        sparse.stride(0),
        logits.stride(0),
        NEXT_N=next_n,
        HEADS=heads,
        DIM=dim,
        PAGE=128,
        PAGES=cache.shape[0],
        TABLE_COLS=tables.shape[1],
        WIDTH=width,
        MAX_LEN=max_model_len,
        BLOCK=64,
        num_warps=4,
        num_stages=2,
    )
    # The score kernel defines each live prefix, including its final tile padding.
    torch.ops._C.persistent_topk(logits, ends, columns, topk_workspace, 512, width)
    _sparse_topk_remap_kernel[(rows,)](
        columns,
        columns.stride(0),
        sparse,
        sparse.stride(0),
        starts,
        starts.stride(0),
        output,
        output.stride(0),
        512,
        width,
        SBK=8,
        K_POW2=512,
    )
