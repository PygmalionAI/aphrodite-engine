#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace aphrodite {

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
    // x * sigmoid(x)
    return (T) (((float) x) / (1.0f + expf((float) - x)));
}

template<typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,             // [num_tokens, d]
    const scalar_t* __restrict__ input,     // [num_tokens, 2, d]
    const int d) {
    const itn token_idx = blockIdx.x;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
        const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
        const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
        out[token_idx * d + idx] = silu(x) * y;
    }
}

} // namespace aphrodite

void silu_and_mul(
    torch::Tensor& out,             // [num_tokens, d]
    torch::Tensor& input)           // [num_tokens, 2 * d]
{
    int num_tokens = input.size(0);
    int d = input.size(1) / 2;

    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "silu_and_mul_kernel",
        [&] {
        aphrodite::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            d);
        });
}
