// adapted from https://github.com/rusty1s/pytorch_bincount
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

namespace aphrodite {
template <typename scalar_t>
__global__ void bincount_kernel(scalar_t *__restrict__ src, int32_t *out,
                                     size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    atomicAdd(out + (ptrdiff_t)src[i], 1);
  }
}
}

void aphrodite_bincount(torch::Tensor src, torch::Tensor out) {
    // the PyTorch bincount is not compatible with CUDA graph
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(
    src.scalar_type(), "bincount_kernel", [&] {
    aphrodite::bincount_kernel<scalar_t><<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
        src.data<scalar_t>(), out.data<int32_t>(), src.numel());
  });
}
