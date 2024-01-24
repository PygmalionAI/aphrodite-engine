#include <cstdint>
#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const bool enable_fp8_kv_cache);

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const bool enable_fp8_kv_cache);

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void fused_add_rms_norm(
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon);

void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox);

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

// The AWQ kernels are only available on CUDA
#ifndef USE_ROCM
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

void marlin_gemm(
  const torch::Tensor& input,
  const torch::Tensor& weights,
        torch::Tensor& output,
  const torch::Tensor& scales,
        torch::Tensor& workspace);

at::Tensor e8p_mm_origorder(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& CB);

void decompress_e8p_origorder(
    torch::Tensor YIs,
    torch::Tensor CB,
    torch::Tensor &Y
);
#endif

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

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

void aphrodite_bincount(
  torch::Tensor src,
  torch::Tensor out);
  