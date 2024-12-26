#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"
#include "quantization/quant_ops.h"

#include <torch/library.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Aphrodite custom ops

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, float k_scale, float v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, float k_scale, float v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  // prepare_inputs advance_step
  ops.def(
      "advance_step(int num_seqs, int num_queries, int block_size, "
      "Tensor! input_tokens, Tensor sampled_token_ids, "
      "Tensor! input_positions, Tensor! seq_lens, Tensor! slot_mapping, "
      "Tensor block_tables) -> ()");
  ops.impl("advance_step", torch::kCUDA, &advance_step);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  ops.def(
      "batched_rotary_embedding(Tensor positions, Tensor! query,"
      "                         Tensor! key, int head_size,"
      "                         Tensor cos_sin_cache, bool is_neox,"
      "                         int rot_dim,"
      "                         Tensor cos_sin_cache_offsets) -> ()");
  ops.impl("batched_rotary_embedding", torch::kCUDA, &batched_rotary_embedding);

  // Quantization ops
#ifndef USE_ROCM
  // Quantized GEMM for AQLM.
  ops.def(
      "aqlm_gemm(Tensor input, Tensor codes, Tensor codebooks, "
      "Tensor scales, int[] codebook_partition_sizes, Tensor? bias) "
      "-> Tensor");
  ops.impl("aqlm_gemm", torch::kCUDA, &aqlm_gemm);

  // Decompression method for AQLM.
  ops.def(
      "aqlm_dequant(Tensor codes, Tensor codebooks, "
      "int[] codebook_partition_sizes) -> Tensor");
  ops.impl("aqlm_dequant", torch::kCUDA, &aqlm_dequant);

  // Quantized GEMM for AWQ.
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, int split_k_iters) -> Tensor");
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  // Dequantization for AWQ.
  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, int split_k_iters, int thx, int thy) -> Tensor");
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // Dequantization for GGML.
  ops.def("ggml_dequantize(Tensor W, int type, int m, int n) -> Tensor");
  ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, int row) "
      "-> Tensor");
  ops.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  // mmq kernel for GGML.
  ops.def("ggml_mul_mat_a8(Tensor W, Tensor X, int type, int row) -> Tensor");
  ops.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA
  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, int size_m, int size_n, int size_k) -> Tensor");
  ops.impl("marlin_gemm", torch::kCUDA, &marlin_gemm);

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "__torch__.torch.classes._core_C.ScalarType b_q_type, "
      "int size_m, int size_n, int size_k) -> Tensor");
  ops.impl("gptq_marlin_24_gemm", torch::kCUDA, &gptq_marlin_24_gemm);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor b_zeros, Tensor g_idx, Tensor perm, Tensor workspace, "
      "__torch__.torch.classes._core_C.ScalarType b_q_type, "
      "int size_m, int size_n, int size_k, bool is_k_full, "
      "bool has_zp, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  ops.impl("gptq_marlin_gemm", torch::kCUDA, &gptq_marlin_gemm);

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
  ops.impl("gptq_marlin_repack", torch::kCUDA, &gptq_marlin_repack);
  ops.impl("gptq_marlin_repack", torch::kMeta, &gptq_marlin_repack_meta);

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");
  ops.impl("awq_marlin_repack", torch::kCUDA, &awq_marlin_repack);
  ops.impl("awq_marlin_repack", torch::kMeta, &awq_marlin_repack_meta);

  // fp8_marlin Optimized Quantized GEMM for FP8 weight-only.
  ops.def(
      "fp8_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, int num_bits, int size_m, int size_n, "
      "int size_k) -> Tensor");
  ops.impl("fp8_marlin_gemm", torch::kCUDA, &fp8_marlin_gemm);

  #ifndef _WIN32
  // marlin_qqq_gemm for QQQ.
  ops.def(
      "marlin_qqq_gemm(Tensor a, Tensor b_q_weight, "
      "Tensor s_tok, Tensor s_ch, Tensor s_group, "
      "Tensor! workspace, int size_m, int size_n, "
      "int size_k) -> Tensor");
  ops.impl("marlin_qqq_gemm", torch::kCUDA, &marlin_qqq_gemm);

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm_azp", torch::kCUDA, &cutlass_scaled_mm_azp);

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def("machete_supported_schedules", &machete::supported_schedules);
  ops.def(
      "machete_gemm(Tensor A, Tensor B,"
      "             __torch__.torch.classes._core_C.ScalarType btype,"
      "             Tensor? scales, Tensor? zeros, int? group_size,"
      "             Tensor? C, float? alpha, float? beta, str? schedule)"
      "-> Tensor");
  ops.impl("machete_gemm", torch::kCUDA, &machete::gemm);
  ops.def(
      "machete_prepack_B(Tensor B,"
      "                  __torch__.torch.classes._core_C.ScalarType btype)"
      "-> Tensor");
  ops.impl("machete_prepack_B", torch::kCUDA, &machete::prepack_B);

  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.impl("permute_cols", torch::kCUDA, &permute_cols);

  #endif

  // QuIP# GEMV
  ops.def("quip_gemv(Tensor A, Tensor B, Tensor CB) -> Tensor",
          &e8p_mm_origorder);
  ops.impl("quip_gemv", torch::kCUDA, &e8p_mm_origorder);

  // QuIP# Decompress
  ops.def("quip_decompress(Tensor YIs, Tensor CB, Tensor Y) -> ()",
          &decompress_e8p_origorder);
  ops.impl("quip_decompress", torch::kCUDA, &decompress_e8p_origorder);

  // fp6_llm
  ops.def(
      "fp_eXmY_linear_forward_cuda(int EXPONENT, int MANTISSA,"
      "                            Tensor _in_feats, Tensor _weights,"
      "                            Tensor _scales, int splitK=1) -> Tensor");
  ops.impl("fp_eXmY_linear_forward_cuda", torch::kCUDA,
           &fp_eXmY_linear_forward_cuda);

  // Sampling Kernels
  ops.def(
      "sampling_from_probs(Tensor probs, Tensor uniform_samples, bool "
      "deterministic) -> Tensor",
      &sampling_from_probs);
  ops.impl("sampling_from_probs", torch::kCUDA, &sampling_from_probs);

  ops.def(
      "top_k_sampling_from_probs(Tensor probs, Tensor uniform_samples,"
      "                          Tensor? maybe_top_k_arr, int top_k_val,"
      "                          bool deterministic) -> Tensor[]",
      &top_k_sampling_from_probs);
  ops.impl("top_k_sampling_from_probs", torch::kCUDA,
           &top_k_sampling_from_probs);

  ops.def(
      "min_p_sampling_from_probs(Tensor probs, Tensor uniform_samples,"
      "                          Tensor? maybe_min_p_arr, float min_p_val,"
      "                          bool deterministic) -> Tensor[]",
      &min_p_sampling_from_probs);
  ops.impl("min_p_sampling_from_probs", torch::kCUDA,
           &min_p_sampling_from_probs);

  ops.def(
      "top_p_sampling_from_probs(Tensor probs, Tensor uniform_samples,"
      "                          Tensor? maybe_top_p_arr, float top_p_val,"
      "                          bool deterministic) -> Tensor[]",
      &top_p_sampling_from_probs);
  ops.impl("top_p_sampling_from_probs", torch::kCUDA,
           &top_p_sampling_from_probs);

  ops.def(
      "top_k_top_p_sampling_from_probs(Tensor probs, Tensor uniform_samples,"
      "                          Tensor? maybe_top_k_arr, float top_k_val,"
      "                          Tensor? maybe_top_p_arr, float top_p_val,"
      "                          bool deterministic) -> Tensor[]",
      &top_k_top_p_sampling_from_probs);
  ops.impl("top_k_top_p_sampling_from_probs", torch::kCUDA,
           &top_k_top_p_sampling_from_probs);

  ops.def(
      "top_k_renorm_prob(Tensor probs, Tensor? maybe_top_k_arr, int top_k_val) "
      "-> Tensor",
      &top_k_renorm_prob);
  ops.impl("top_k_renorm_prob", torch::kCUDA, &top_k_renorm_prob);

  ops.def(
      "top_p_renorm_prob(Tensor probs, Tensor? maybe_top_p_arr, float "
      "top_p_val) "
      "-> Tensor",
      &top_p_renorm_prob);
  ops.impl("top_p_renorm_prob", torch::kCUDA, &top_p_renorm_prob);

  ops.def(
      "top_k_mask_logits(Tensor logits, Tensor? maybe_top_k_arr, int "
      "top_k_val) -> Tensor",
      &top_k_mask_logits);
  ops.impl("top_k_mask_logits", torch::kCUDA, &top_k_mask_logits);

#endif

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, int bit) "
      "-> Tensor");
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Quantized GEMM for SqueezeLLM.
  ops.def(
      "squeezellm_gemm(Tensor vec, Tensor mat, Tensor! mul, Tensor "
      "lookup_table) -> ()");
  ops.impl("squeezellm_gemm", torch::kCUDA, &squeezellm_gemm);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! out, Tensor input, Tensor scale) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! out, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  ops.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  ops.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // Compute int8 quantized tensor for given scaling factor.
  /*
    Implementation:
    void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const&
    input, torch::Tensor const& scale);
  */
  ops.def(
      "static_scaled_int8_quant(Tensor! out, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  /*
    Implementation:
    void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const&
    input, torch::Tensor& scales);
  */
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);
#ifndef USE_ROCM
  // Mamba kernels
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? index_, Tensor(a! -> *)? x) -> Tensor(a)[]");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  ops.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation) -> Tensor");
  ops.impl("causal_conv1d_update", torch::kCUDA, &causal_conv1d_update);

  ops.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor? seq_idx_,"
      "Tensor? seq_pos_idx_,"
      "Tensor? initial_states_,"
      "Tensor? final_states_out_,"
      "bool silu_activation) -> Tensor");
  ops.impl("causal_conv1d_fwd", torch::kCUDA, &causal_conv1d_fwd);
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.def(
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  float k_scale, float v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        float k_scale, float v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  // Convert the key and value cache to fp8 data type.
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  cache_ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}

#ifndef USE_ROCM
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def(
      "init_custom_ar(Tensor meta, Tensor rank_data, "
      "str[] handles, int[] offsets, int rank, "
      "bool full_nvlink) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "should_custom_ar(Tensor inp, int max_size, int world_size, "
      "bool full_nvlink) -> bool");
  custom_ar.impl("should_custom_ar", torch::kCUDA, &should_custom_ar);

  custom_ar.def("all_reduce_reg(int fa, Tensor inp, Tensor! out) -> ()");
  custom_ar.impl("all_reduce_reg", torch::kCUDA, &all_reduce_reg);

  custom_ar.def(
      "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> "
      "()");
  custom_ar.impl("all_reduce_unreg", torch::kCUDA, &all_reduce_unreg);

  custom_ar.def("dispose", &dispose);

  custom_ar.def("meta_size", &meta_size);

  custom_ar.def(
      "register_buffer(int fa, Tensor t, str[] handles, "
      "int[] offsets) -> ()");
  custom_ar.impl("register_buffer", torch::kCUDA, &register_buffer);

  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);

  custom_ar.def("register_graph_buffers", &register_graph_buffers);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)