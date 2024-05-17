#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cutlass_kernels/fpA_intB_gemm_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("w8_a16_gemm", &w8_a16_gemm_forward_cuda, "Weight only gemm");
  m.def("w8_a16_gemm_", &w8_a16_gemm_forward_cuda_, "Weight only gemm inplace");
}