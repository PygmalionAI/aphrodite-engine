#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "quant_utils.cuh"
#include "reduction_utils.cuh"

namespace aphrodite {

// TODO: Further optimize this kernel.
template <typename scalar_t>
__global__ void
rms_norm_kernel(scalar_t *__restrict__ out,         // [num_tokens, hidden_size]
                const scalar_t *__restrict__ input, // [num_tokens, hidden_size]
                const scalar_t *__restrict__ weight, // [hidden_size]
                const float epsilon, const int num_tokens,
                const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

template <typename T>
__global__ void RMSLayerNorm(const T *__restrict input,
                             const T *__restrict gamma, int8_t *output,
                             const float layernorm_eps, int m, int n) {
  // layernorm module in the T5 style No bias and no subtraction of mean.
  const int tid = threadIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    // float diff = (float)(ldg(&input[blockIdx.x * n + i]));
    float diff = (float)(input[blockIdx.x * n + i]);
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / (float)n + layernorm_eps);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        // float_to_int8_rn((((float)input[blockIdx.x * n + i]) * s_variance) *
        //                  (float)(ldg(&gamma[i])));
        float_to_int8_rn((((float)input[blockIdx.x * n + i]) * s_variance) *
                         (float)(gamma[i]));
  }
}

template <typename T>
void invokeRMSLayerNorm(int8_t *out, const T *input, const T *gamma,
                        //   const T*     beta,
                        const float layernorm_eps, const int m, const int n,
                        cudaStream_t stream) {
  // if (beta != nullptr) {
  //     invokeGeneralLayerNorm(out, input, gamma, beta, layernorm_eps, m, n,
  //     (float*)nullptr, 0, stream); return;
  // }

  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
      Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if (n % 32 != 0) {
    block.x = 1024;
  }

  block.x =
      block.x / (4 / sizeof(T)); // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  RMSLayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, out, layernorm_eps,
                                              m, n); // For gpt-3
}

} // namespace aphrodite

void rms_norm(torch::Tensor &out,    // [num_tokens, hidden_size]
              torch::Tensor &input,  // [num_tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon) {
  int num_tokens = input.size(0);
  int hidden_size = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  APHRODITE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    aphrodite::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
  });
}

void invoke_rms_norm_quant(torch::Tensor &out,   // [num_tokens, hidden_size]
                           torch::Tensor &input, // [num_tokens, hidden_size]
                           torch::Tensor &gamma, // [hidden_size]
                           float epsilon) {
  int m = input.size(0);
  int n = input.size(1);
  dim3 grid(m);
  dim3 block(min(n, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  APHRODITE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "invokeRMSLayerNorm", [&] {
    aphrodite::RMSLayerNorm<scalar_t><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(), gamma.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
         epsilon, m, n);
  });
}