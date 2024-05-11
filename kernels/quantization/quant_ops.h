#pragma once

#include <torch/extension.h>

#ifndef USE_ROCM
// AQLM
torch::Tensor aqlm_gemm(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& codebooks,
  const torch::Tensor& scales,
  const torch::Tensor& codebook_partition_sizes,
  const std::optional<torch::Tensor>& bias
);

// AWQ
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

torch::Tensor awq_dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy);

torch::Tensor awq_group_gemm(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    torch::Tensor _topk_weights,
    torch::Tensor _sorted_token_ids_ptr,
    torch::Tensor _expert_ids_ptr,
    torch::Tensor _num_tokens_post_padded,
    bool mul_weights,
    int split_k_iters);
#endif

// ExLlamav2
torch::Tensor exl2_gemm(
    torch::Tensor a,
    uintptr_t b
);

uintptr_t make_q_matrix(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map
);

#ifndef USE_ROCM
// GGUF
torch::Tensor ggml_dequantize(
    torch::Tensor X,
    int8_t type,
    int64_t m,
    int64_t n
);

torch::Tensor ggml_mul_mat_vec(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int8_t type,
    int64_t m
);

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int8_t type,
    int64_t row
);

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int8_t type,
    int64_t row
);
#endif

// GPTQ
torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama,
  int bit);

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm,
  int bit);

torch::Tensor group_gptq_gemm(
    torch::Tensor a,
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids_ptr,
    torch::Tensor expert_ids_ptr,
    torch::Tensor num_tokens_post_padded,
    bool mul_weights,
    bool use_exllama
);

torch::Tensor dequant_gptq(
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    int bits,
    bool use_exllama
);

#ifndef USE_ROCM
// Marlin
torch::Tensor marlin_gemm(
    torch::Tensor& a,
    torch::Tensor& b_q_weight,
    torch::Tensor& b_scales,
    torch::Tensor& workspace,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k);

// QuIP#
at::Tensor e8p_mm_origorder(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& CB);

void decompress_e8p_origorder(
    torch::Tensor YIs,
    torch::Tensor CB,
    torch::Tensor &Y
);

// SmoothQuant+
torch::Tensor autoquant_s4_f16_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scales_zeros);

void autoquant_convert_s4_k_m8(
  torch::Tensor _weight_dest,
  torch::Tensor _quant_scales_zeros_dest,
  torch::Tensor _workspace,
  torch::Tensor _quant_weight_src,
  torch::Tensor _quant_scales,
  torch::Tensor _quant_zeros,
  int m,
  int k,
  int group_size);
#endif

// SqueezeLLM
void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);
