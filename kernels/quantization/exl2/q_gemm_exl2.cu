/*
 * Adapted from https://github.com/turboderp/exllamav2
 * Copyright (c) 2024 turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "q_matrix.cuh"
#include "matrix_view.cuh"
#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"
#include "q_gemm_kernel.cuh"

namespace aphrodite {
namespace exl2 {

#define MAX_Q_GEMM_ROWS 32
#define EXL2_BLOCK_KN_SIZE 64
#define EXL2_BLOCK_M_SIZE_MAX 8
#define EXL2_MAX_GROUPS_IN_BLOCK (EXL2_BLOCK_KN_SIZE / 32)

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

void gemm_half_q_half_cuda_part
(
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear
)
{
    {
        dim3 blockDim, gridDim;
        blockDim.x = EXL2_BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(size_n, EXL2_BLOCK_KN_SIZE * 4);
        gridDim.y = DIVIDE(size_m, m_count);
        gridDim.z = DIVIDE(size_k, EXL2_BLOCK_KN_SIZE);

        fp_gemm_half_q_half_kernel kernel = pick_gemm_half_q_half_kernel(m_count);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        kernel<<<gridDim, blockDim, 0, stream>>>
        (
            a,
            b->cuda_q_weight,
            b->cuda_q_scale,
            b->cuda_q_scale_max,
            c,
            size_m,
            size_n,
            size_k,
            b->groups,
            b->cuda_q_group_map,
            b->cuda_q_perm,
            b->rows_8,
            b->rows_6,
            b->rows_5,
            b->rows_4,
            b->rows_3,
            b->rows_2,
            clear
        );
    }

}

void gemm_half_q_half_cuda
(
    cublasHandle_t cublas_handle,
    const half* a,
    QMatrix* b,
    half* c,
    int size_m,
    int size_n,
    int size_k,
    bool clear,
    half* temp_dq
)
{
    if (size_m > MAX_Q_GEMM_ROWS)
    {
        // Reconstruct FP16 matrix, then cuBLAS
        b->reconstruct(temp_dq);

        //cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

        const half alpha = __float2half(1.0f);
        const half beta = clear ? __float2half(0.0f) : __float2half(1.0f);
        cublasHgemm(cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    size_n, size_m, size_k,
                    &alpha, temp_dq, size_n,
                            a,       size_k,
                    &beta,  c,       size_n);
    }
    else
    {
        // Quantized matmul

        int block_m_size_max = EXL2_BLOCK_M_SIZE_MAX;
        int max_chunks = size_m / block_m_size_max;
        int last_chunk = max_chunks * block_m_size_max;
        int last_chunk_size = size_m - last_chunk;

        if (max_chunks)
        {
            gemm_half_q_half_cuda_part(a, b, c, last_chunk, size_n, size_k, block_m_size_max, clear);
        }

        if (last_chunk_size)
        {
            gemm_half_q_half_cuda_part(a + last_chunk * size_k, b, c + last_chunk * size_n, last_chunk_size, size_n, size_k, last_chunk_size, clear);
        }
    }
}

}  // namespace exl2
}  // namespace aphrodite

torch::Tensor exl2_gemm
(
    torch::Tensor a,
    uintptr_t b
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    aphrodite::exl2::QMatrix* qm = reinterpret_cast<aphrodite::exl2::QMatrix*> (b);

    auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
    at::Tensor c = torch::empty({a.size(0), qm->width}, options);
    at::Tensor temp_dq = torch::empty({a.size(1), qm->width}, options);

    aphrodite::exl2::gemm_half_q_half_cuda
    (
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        qm,
        (half*) c.data_ptr(),
        c.size(0),  // m
        c.size(1),  // n
        a.size(1),  // k
        true,
        (half*) temp_dq.data_ptr()
    );
    return c;
}

uintptr_t make_q_matrix
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q_weight));
    int device = q_weight.device().index();
    int width = q_weight.size(1);
    int groups = q_scale.size(0);
    int height = q_invperm.size(0);

    aphrodite::exl2::QMatrix* m = new aphrodite::exl2::QMatrix
    (
        device,
        height,
        width,
        groups,
        (uint32_t*) q_weight.data_ptr(),
        (uint16_t*) q_perm.data_ptr(),
        (uint16_t*) q_invperm.data_ptr(),
        (uint32_t*) q_scale.data_ptr(),
        (half*) q_scale_max.data_ptr(),
        (uint16_t*) q_groups.data_ptr(),
        (uint16_t*) q_group_map.data_ptr()
    );
    return reinterpret_cast<uintptr_t>(m);
}