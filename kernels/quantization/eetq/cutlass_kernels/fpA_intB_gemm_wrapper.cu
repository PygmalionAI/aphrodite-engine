#include <torch/extension.h>
#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include "fpA_intB_gemm_wrapper.h"
#include "fpA_intB_gemm.h"
#include "cuda_utils.h"
#include "weightOnlyBatchedGemv/enabled.h"
#include "weightOnlyBatchedGemv/kernelLauncher.h"
#include "torch_utils.h"

#include <vector>

namespace ft = fastertransformer;

int getWorkspaceSize(const int m, const int n, const int k)
{
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = (m + 31) / 32;
    const int max_grid_n = (n + 127) / 128;
    const int split_k_limit = 7;
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return max_grid_m * max_grid_n * split_k_limit * 4;
}

std::vector<torch::Tensor>

torch::Tensor w8_a16_gemm_forward_cuda(torch::Tensor &input,
                                       torch::Tensor &weight,
                                       torch::Tensor &scale)
{
    c10::cuda::CUDAGuard device_guard(input.device());
    // TORCH_CHECK(input.dim() == 3 || input.dim() == 2, "Invalid input dim: ", input.dim());
    const int m = input.dim() == 2 ? input.size(0) : input.size(0) * input.size(1);
    const int k = input.size(-1);
    const int n = weight.size(-1);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = input.dim() == 2 ? torch::empty({m, n}, options) : torch::empty({input.size(0), input.size(1), n}, options);
    const ft::half *input_ptr = reinterpret_cast<ft::half *>(input.data_ptr());
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight.data_ptr());
    const ft::half *scale_ptr = reinterpret_cast<ft::half *>(scale.data_ptr());
    ft::half *output_ptr = reinterpret_cast<ft::half *>(output.data_ptr());
    // const int max_size = std::max(n, k);
    // size_t workspace_size = getWorkspaceSize(m, max_size, max_size);
    // void *ptr = nullptr;
    // char *workspace_ptr = workspace_size > 0 ? (char *)cudaMalloc((void **)&ptr, workspace_size) : nullptr;
    const bool use_cuda_kernel = m <= SMALL_M_FAST_PATH;
    // const bool use_cuda_kernel = false; 
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(use_cuda_kernel){
        tensorrt_llm::kernels::WeightOnlyActivationType weight_only_act_type = tensorrt_llm::kernels::WeightOnlyActivationType::FP16;
        tensorrt_llm::kernels::WeightOnlyQuantType weight_only_quant_type = tensorrt_llm::kernels::WeightOnlyQuantType::Int8b;
        tensorrt_llm::kernels::WeightOnlyParams params{weight_ptr, reinterpret_cast<const uint8_t *>(scale.data_ptr()), nullptr,
            reinterpret_cast<half *>(input.data_ptr()), nullptr, nullptr, reinterpret_cast<half *>(output.data_ptr()), m, n, k, 0, weight_only_quant_type,
            tensorrt_llm::kernels::WeightOnlyType::PerChannel,
            tensorrt_llm::kernels::WeightOnlyActivationFunctionType::Identity, weight_only_act_type};
        tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, stream);
    }
    else
        ft::gemm_fp16_int(
            input_ptr,
            weight_ptr,
            scale_ptr,
            output_ptr,
            m, n, k,
            nullptr,
            0,
            stream);
    return output;
}


torch::Tensor w8_a16_gemm_forward_cuda_(torch::Tensor &input,
                                        torch::Tensor &weight,
                                        torch::Tensor &scale,
                                        torch::Tensor &output,
                                        const int m,
                                        const int n,
                                        const int k)
{
    c10::cuda::CUDAGuard device_guard(input.device());

    const ft::half *input_ptr = reinterpret_cast<ft::half *>(input.data_ptr());
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight.data_ptr());
    const ft::half *scale_ptr = reinterpret_cast<ft::half *>(scale.data_ptr());
    ft::half *output_ptr = reinterpret_cast<ft::half *>(output.data_ptr());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ft::gemm_fp16_int(
        input_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        m, n, k,
        nullptr,
        0,
        stream);
    return output;
}