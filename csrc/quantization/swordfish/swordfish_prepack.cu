// Packs GPTQ weights directly into Swordfish ABI v1, including its Marlin
// 16x64 in-tile permutation and optional activation-order row sort.

#include <optional>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "torch_utils.h"
#include "swordfish_abi.cuh"

namespace swordfish {

namespace {

template <int kBits, bool kHasPerm>
__global__ void pack_gptq_to_blocks(const uint32_t* __restrict__ weights,
                                    const int32_t* __restrict__ perm,
                                    uint32_t* __restrict__ out, int64_t num_kb,
                                    int64_t size_n) {
  constexpr int kPack = 32 / kBits;
  constexpr int kTileWords = 1024 / kPack;
  constexpr int kWords = 4 * kTileWords;
  const int64_t nb = blockIdx.x;
  const int64_t kb = blockIdx.y;
  for (int w = threadIdx.x; w < kWords; w += blockDim.x) {
    const int slice = w / kTileWords;
    const int word = w % kTileWords;
    uint32_t packed = 0;
#pragma unroll
    for (int bit = 0; bit < kPack; ++bit) {
      const int interleave = 2 * (bit % (kPack / 2)) + bit / (kPack / 2);
      const int p = word * kPack + interleave;
      const int lane = p / 32;
      const int r = p % 4;
      int64_t row = kb * 64 + slice * 16 + 2 * (lane % 4) + r % 2 + 8 * (r / 2);
      if constexpr (kHasPerm) row = perm[row];
      const int64_t col =
          nb * 64 + 16 * ((p / 8) % 4) + lane / 4 + 8 * ((p / 4) % 2);
      const uint32_t source = weights[(row / kPack) * size_n + col];
      const uint32_t value =
          (source >> (kBits * (row % kPack))) & ((1u << kBits) - 1);
      packed |= value << (kBits * bit);
    }
    out[(nb * num_kb + kb) * kWords + w] = packed;
  }
}

}  // namespace

torch::stable::Tensor swordfish_prepack_B(
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& perm, int64_t size_k,
    int64_t size_n, int64_t num_bits) {
  STD_TORCH_CHECK(shape_ok(size_k, size_n), "swordfish ABI v1 requires K % ",
                  kBlockK, " == 0 and N % ", kBlockN, " == 0; got K=", size_k,
                  " N=", size_n, " (v1 tail policy: reject)");
  STD_TORCH_CHECK(num_bits == 4 || num_bits == 8,
                  "swordfish supports 4-bit and 8-bit weights");

  const int64_t pack_factor = 32 / num_bits;
  STD_TORCH_CHECK(
      b_q_weight.is_cuda() && b_q_weight.is_contiguous() &&
          b_q_weight.scalar_type() == torch::headeronly::ScalarType::Int &&
          b_q_weight.dim() == 2 && b_q_weight.size(0) == size_k / pack_factor &&
          b_q_weight.size(1) == size_n,
      "Swordfish expects contiguous CUDA int32 GPTQ weights [K/pack, N]");
  const bool has_perm = perm.has_value() && perm->numel() != 0;
  if (has_perm) {
    STD_TORCH_CHECK(
        perm->is_cuda() &&
            perm->get_device_index() == b_q_weight.get_device_index() &&
            perm->is_contiguous() &&
            perm->scalar_type() == torch::headeronly::ScalarType::Int &&
            perm->dim() == 1 && perm->size(0) == size_k,
        "Swordfish permutation must be contiguous CUDA int32 [K] on the weight "
        "device");
  }
  const int32_t device_index = b_q_weight.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  const int64_t nb = num_blocks_n(size_n);
  const int64_t kb = num_blocks_k(size_k);
  const int64_t words = num_bits == 8 ? kBlockInt32_8 : kBlockInt32;
  torch::stable::Tensor out =
      torch::stable::empty({nb, kb, words}, torch::headeronly::ScalarType::Int,
                           std::nullopt, b_q_weight.device());

  const auto* source =
      reinterpret_cast<const uint32_t*>(b_q_weight.const_data_ptr());
  const auto* indices =
      has_perm ? reinterpret_cast<const int32_t*>(perm->const_data_ptr())
               : nullptr;
  auto* destination = reinterpret_cast<uint32_t*>(out.mutable_data_ptr());
  dim3 grid(nb, kb);
  if (num_bits == 8) {
    if (has_perm)
      pack_gptq_to_blocks<8, true>
          <<<grid, 256, 0, stream>>>(source, indices, destination, kb, size_n);
    else
      pack_gptq_to_blocks<8, false>
          <<<grid, 256, 0, stream>>>(source, indices, destination, kb, size_n);
  } else {
    if (has_perm)
      pack_gptq_to_blocks<4, true>
          <<<grid, 256, 0, stream>>>(source, indices, destination, kb, size_n);
    else
      pack_gptq_to_blocks<4, false>
          <<<grid, 256, 0, stream>>>(source, indices, destination, kb, size_n);
  }

  return out;
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_prepack_B", TORCH_BOX(&swordfish::swordfish_prepack_B));
}
