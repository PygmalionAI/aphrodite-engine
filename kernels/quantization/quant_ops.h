#pragma once

#include <torch/library.h>
#include "core/scalar_type.hpp"

#ifndef USE_ROCM
// AQLM
torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const std::vector<int64_t>& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias);

torch::Tensor aqlm_dequant(
    const torch::Tensor& codes, const torch::Tensor& codebooks,
    const std::vector<int64_t>& codebook_partition_sizes);

// AWQ
torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor awq_group_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, torch::Tensor _topk_weights,
                             torch::Tensor _sorted_token_ids_ptr,
                             torch::Tensor _expert_ids_ptr,
                             torch::Tensor _num_tokens_post_padded,
                             bool mul_weights, int64_t split_k_iters);
#endif

// GPTQ
torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

torch::Tensor group_gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_gptq_qzeros,
                              torch::Tensor b_gptq_scales,
                              torch::Tensor b_g_idx, torch::Tensor topk_weights,
                              torch::Tensor sorted_token_ids_ptr,
                              torch::Tensor expert_ids_ptr,
                              torch::Tensor num_tokens_post_padded,
                              bool mul_weights, bool use_exllama);

torch::Tensor dequant_gptq(torch::Tensor b_q_weight,
                           torch::Tensor b_gptq_qzeros,
                           torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                           int64_t bits, bool use_exllama);

#ifndef USE_ROCM
// Marlin
torch::Tensor marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                          torch::Tensor& b_scales, torch::Tensor& workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);

torch::Tensor gptq_marlin_24_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                                  torch::Tensor& b_meta,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace,
                                  aphrodite::ScalarTypeTorchPtr const& b_q_type,
                                  int64_t size_m, int64_t size_n,
                                  int64_t size_k);

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace,
                               aphrodite::ScalarTypeTorchPtr const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce);

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits);

torch::Tensor fp8_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                              torch::Tensor& b_scales, torch::Tensor& workspace,
                              int64_t num_bits, int64_t size_m, int64_t size_n,
                              int64_t size_k);

// GGUF
torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

// QuIP#
at::Tensor e8p_mm_origorder(const at::Tensor& A, const at::Tensor& B,
                            const at::Tensor& CB);

void decompress_e8p_origorder(torch::Tensor YIs, torch::Tensor CB,
                              torch::Tensor& Y);

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp(torch::Tensor& out, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           c10::optional<torch::Tensor> const& azp,
                           c10::optional<torch::Tensor> const& bias);

torch::Tensor marlin_qqq_gemm(torch::Tensor const& a,
                              torch::Tensor const& b_q_weight,
                              torch::Tensor const& s_tok,
                              torch::Tensor const& s_ch,
                              torch::Tensor const& s_group,
                              torch::Tensor& workspace, int64_t size_m,
                              int64_t size_n, int64_t size_k);

torch::Tensor fp_eXmY_linear_forward_cuda(int64_t EXPONENT, int64_t MANTISSA,
                                          torch::Tensor _in_feats,
                                          torch::Tensor _weights,
                                          torch::Tensor _scales,
                                          int64_t splitK = 1);

#endif

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales);

// SqueezeLLM
void squeezellm_gemm(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                     torch::Tensor lookup_table);

// FP8
void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                             torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    c10::optional<torch::Tensor> const& scale_ub);

// flute
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include "cute/numeric/integral_constant.hpp"

template <typename SMs, typename T, typename TQ, typename T2, typename NumBits,
          typename GroupSize>
void _qgemm(int M, int N, int K, int P, const T* const __restrict__ A,
            const TQ* const __restrict__ Q, T* __restrict__ D,
            const T* const __restrict__ S, const T* const __restrict__ QM,
            const T2* const __restrict__ QM2, void* __restrict__ workspace,
            const cudaStream_t stream);

template <typename SMs, typename T, typename TQ, typename T2, typename NumBits,
          typename GroupSize>
void _qgemm_raw(int M, int N, int K, int P, const T* const __restrict__ A,
                const TQ* const __restrict__ Q, T* __restrict__ D,
                const T* const __restrict__ S, const T* const __restrict__ QM,
                const T2* const __restrict__ QM2, void* __restrict__ workspace,
                const int template_id, const cudaStream_t stream);

template <typename SMs, typename T, typename NumBits, typename GroupSize>
void qgemm(const at::Tensor& input, const at::Tensor& weight,
           at::Tensor& output, const at::Tensor& scales,
           const at::Tensor& table, const at::Tensor& table2,
           at::Tensor& workspace, const cudaStream_t stream) {
  using namespace cute;
  using TQ = cute::uint16_t;
  using T2 = conditional_t<is_same_v<T, half_t>, __half2, __nv_bfloat162>;

  _qgemm<SMs, T, TQ, T2, NumBits,
         GroupSize>(output.size(0),  // M
                    output.size(1),  // N
                    input.size(1),   // K
                    weight.size(0),  // P
                    reinterpret_cast<const T*>(input.data_ptr()),
                    reinterpret_cast<const TQ*>(weight.data_ptr()),
                    reinterpret_cast<T*>(output.data_ptr()),
                    reinterpret_cast<const T*>(scales.data_ptr()),
                    reinterpret_cast<const T*>(table.data_ptr()),
                    reinterpret_cast<const T2*>(table2.data_ptr()),
                    reinterpret_cast<void*>(workspace.data_ptr()), stream);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SMs, typename T, typename NumBits, typename GroupSize>
void qgemm_raw(const at::Tensor& input, const at::Tensor& weight,
               at::Tensor& output, const at::Tensor& scales,
               const at::Tensor& table, const at::Tensor& table2,
               at::Tensor& workspace, const int template_id,
               const cudaStream_t stream) {
  using namespace cute;
  using TQ = cute::uint16_t;
  using T2 = conditional_t<is_same_v<T, half_t>, __half2, __nv_bfloat162>;

  _qgemm_raw<SMs, T, TQ, T2, NumBits,
             GroupSize>(output.size(0),  // M
                        output.size(1),  // N
                        input.size(1),   // K
                        weight.size(0),  // P
                        reinterpret_cast<const T*>(input.data_ptr()),
                        reinterpret_cast<const TQ*>(weight.data_ptr()),
                        reinterpret_cast<T*>(output.data_ptr()),
                        reinterpret_cast<const T*>(scales.data_ptr()),
                        reinterpret_cast<const T*>(table.data_ptr()),
                        reinterpret_cast<const T2*>(table2.data_ptr()),
                        reinterpret_cast<void*>(workspace.data_ptr()),
                        template_id, stream);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SMs>
at::Tensor qgemm_simple(const at::Tensor& input, const at::Tensor& weight,
                        const at::Tensor& scales, const at::Tensor& table,
                        const at::Tensor& table2, at::Tensor& workspace,
                        const cute::int64_t num_bits,
                        const cute::int64_t group_size) {
  // Set the device of this function, primarily used when
  // we have multiple devices in the same process.
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  // Get the current CUDA stream, primarily used
  // to make CUDA Graphs work.
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Squash the batch dimensions of the input tensor with its
  // next-to-last dimensions.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});
  auto output = at::empty(
      {input_2d.size(0), scales.size(0)},
      at::TensorOptions().dtype(input_2d.dtype()).device(input_2d.device()));

#define RUN_QGEMM(T, NUM_BITS, GROUP_SIZE)                                   \
  do {                                                                       \
    qgemm<SMs, T, cute::Int<NUM_BITS>, cute::Int<GROUP_SIZE> >(              \
        input_2d, weight, output, scales, table, table2, workspace, stream); \
  } while (false)

#define RUN_QGEMM_SWITCH_GROUP_SIZE(T, NUM_BITS) \
  do {                                           \
    switch (group_size) {                        \
      case 32:                                   \
        RUN_QGEMM(T, NUM_BITS, 32);              \
        break;                                   \
      case 64:                                   \
        RUN_QGEMM(T, NUM_BITS, 64);              \
        break;                                   \
      case 128:                                  \
        RUN_QGEMM(T, NUM_BITS, 128);             \
        break;                                   \
      case 256:                                  \
        RUN_QGEMM(T, NUM_BITS, 256);             \
        break;                                   \
      default:                                   \
        AT_ERROR("Unsupported `group_size`");    \
    }                                            \
  } while (false)

#define RUN_QGEMM_SWITCH_NUM_BITS_AND_GROUP_SIZE(T) \
  do {                                              \
    switch (num_bits) {                             \
      case 2:                                       \
        RUN_QGEMM_SWITCH_GROUP_SIZE(T, 2);          \
        break;                                      \
      case 3:                                       \
        RUN_QGEMM_SWITCH_GROUP_SIZE(T, 3);          \
        break;                                      \
      case 4:                                       \
        RUN_QGEMM_SWITCH_GROUP_SIZE(T, 4);          \
        break;                                      \
      default:                                      \
        AT_ERROR("Unsupported `num_bits`");         \
    }                                               \
  } while (false)

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "qgemm_simple",
      AT_DISPATCH_CASE(at::ScalarType::Half, [&]() {
        RUN_QGEMM_SWITCH_NUM_BITS_AND_GROUP_SIZE(cute::half_t);
        return;
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]() {
        RUN_QGEMM_SWITCH_NUM_BITS_AND_GROUP_SIZE(cute::bfloat16_t);
        return;
      }));

  auto output_sizes = input_sizes;
  output_sizes.back() = scales.size(0);
  return output.reshape(output_sizes);
}

template <typename SMs>
void qgemm_raw_simple(const at::Tensor& input, const at::Tensor& weight,
                      at::Tensor& output, const at::Tensor& scales,
                      const at::Tensor& table, const at::Tensor& table2,
                      at::Tensor& workspace, const cute::int64_t num_bits,
                      const cute::int64_t group_size,
                      const cute::int64_t template_id) {
  // Set the device of this function, primarily used when
  // we have multiple devices in the same process.
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  // Get the current CUDA stream, primarily used
  // to make CUDA Graphs work.
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define RUN_QGEMM_RAW(T, NUM_BITS, GROUP_SIZE)                                \
  do {                                                                        \
    qgemm_raw<SMs, T, cute::Int<NUM_BITS>, cute::Int<GROUP_SIZE> >(           \
        input, weight, output, scales, table, table2, workspace, template_id, \
        stream);                                                              \
  } while (false)

#define RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, NUM_BITS) \
  do {                                               \
    switch (group_size) {                            \
      case 32:                                       \
        RUN_QGEMM_RAW(T, NUM_BITS, 32);              \
        break;                                       \
      case 64:                                       \
        RUN_QGEMM_RAW(T, NUM_BITS, 64);              \
        break;                                       \
      case 128:                                      \
        RUN_QGEMM_RAW(T, NUM_BITS, 128);             \
        break;                                       \
      case 256:                                      \
        RUN_QGEMM_RAW(T, NUM_BITS, 256);             \
        break;                                       \
      default:                                       \
        AT_ERROR("Unsupported `group_size`");        \
    }                                                \
  } while (false)

#define RUN_QGEMM_RAW_SWITCH_NUM_BITS_AND_GROUP_SIZE(T) \
  do {                                                  \
    switch (num_bits) {                                 \
      case 2:                                           \
        RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 2);          \
        break;                                          \
      case 3:                                           \
        RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 3);          \
        break;                                          \
      case 4:                                           \
        RUN_QGEMM_RAW_SWITCH_GROUP_SIZE(T, 4);          \
        break;                                          \
      default:                                          \
        AT_ERROR("Unsupported `num_bits`");             \
    }                                                   \
  } while (false)

  AT_DISPATCH_SWITCH(
      input.scalar_type(), "qgemm_raw_simple",
      AT_DISPATCH_CASE(at::ScalarType::Half, [&]() {
        RUN_QGEMM_RAW_SWITCH_NUM_BITS_AND_GROUP_SIZE(cute::half_t);
        return;
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]() {
        RUN_QGEMM_RAW_SWITCH_NUM_BITS_AND_GROUP_SIZE(cute::bfloat16_t);
        return;
      }));
}