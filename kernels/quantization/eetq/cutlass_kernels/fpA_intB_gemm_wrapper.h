#include <torch/extension.h>
#include <vector>

#define SMALL_M_FAST_PATH 4

torch::Tensor w8_a16_gemm_forward_cuda(torch::Tensor &input,
                                       torch::Tensor &weight,
                                       torch::Tensor &scale);

torch::Tensor w8_a16_gemm_forward_cuda_(torch::Tensor &input,
                                        torch::Tensor &weight,
                                        torch::Tensor &scale,
                                        torch::Tensor &output,
                                        const int m,
                                        const int n,
                                        const int k);