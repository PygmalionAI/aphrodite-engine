// Swordfish ABI v1 address math, shared by the prepack kernel and both
// mainloops. Single source of truth for the (NB, KB, tile) block-linear
// layout; the in-tile permutation is Marlin's.
#pragma once

#include "swordfish_types.cuh"

namespace swordfish {

// Byte offset of packed block (nb, kb) in a (NB, KB, kBlockBytes) tensor.
__host__ __device__ inline constexpr int64_t block_byte_offset(int64_t nb,
                                                               int64_t kb,
                                                               int64_t num_kb) {
  return (nb * num_kb + kb) * kBlockBytes;
}

// Byte offset of the k16-slice sub-tile `t` (0..3) within a block.
__host__ __device__ inline constexpr int64_t subtile_byte_offset(int t) {
  return int64_t(t) * kSubTileBytes;
}

}  // namespace swordfish
