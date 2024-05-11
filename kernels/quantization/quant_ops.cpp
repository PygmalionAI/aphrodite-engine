#include "quant_ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Aphrodite quantization ops
  pybind11::module quant_ops = m.def_submodule("quant_ops", "Aphrodite custom quant operators");

#ifndef USE_ROCM
  // AQLM
  quant_ops.def("aqlm_gemm", &aqlm_gemm, "Quantized GEMM for AQLM");
  // AWQ
  quant_ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  quant_ops.def("awq_dequantize", &awq_dequantize, "Dequantization for AWQ");
  quant_ops.def("awq_group_gemm", &awq_group_gemm, "Grouped Quantized GEMM for AWQ");
  // GGUF
  quant_ops.def("ggml_dequantize", &ggml_dequantize, "ggml_dequantize");
  quant_ops.def("ggml_mul_mat_vec", &ggml_mul_mat_vec, "ggml_mul_mat_vec");
  quant_ops.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "ggml_mul_mat_vec_a8");
  quant_ops.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "ggml_mul_mat_a8");
  // Marlin
  quant_ops.def("marlin_gemm", &marlin_gemm, "Marlin Optimized Quantized GEMM for GPTQ");
  quant_ops.def("autoquant_convert_s4_k_m8", &autoquant_convert_s4_k_m8, "convert kernel.");
  quant_ops.def("autoquant_s4_f16_gemm", &autoquant_s4_f16_gemm, "weight int4 activation float16 gemm kernel.");
  quant_ops.def("quip_decompress", &decompress_e8p_origorder, "decompress_packed_e8p");
  quant_ops.def("quip_gemv", &e8p_mm_origorder, "e8p_mm_origorder");
#endif
  quant_ops.def("gptq_gemm", &gptq_gemm, "Quantized GEMM for GPTQ");
  quant_ops.def("gptq_shuffle", &gptq_shuffle, "Post processing for GPTQ");
  quant_ops.def("group_gptq_gemm", &group_gptq_gemm, "Grouped Quantized GEMM for GPTQ");
  quant_ops.def("dequant_gptq", &dequant_gptq, "Dequantize gptq weight to half");
  quant_ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");
  quant_ops.def("exl2_make_q_matrix",&make_q_matrix, "preprocess for exl2");
  quant_ops.def("exl2_gemm", &exl2_gemm, "exl2 gemm");
}