#pragma once
#include <assert.h>
#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
#include <cooperative_groups/reduce.h>
#else
#include <cooperative_groups.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <type_traits>

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

template<typename T>
inline __device__ T add(T a, T b) {
    return a + b;
}

template<>
inline __device__ half2 add(half2 a, half2 b) {
    return __hadd2(a, b);
}

template<>
inline __device__ half add(half a, half b) {
    return __hadd(a, b);
}

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
        return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum( T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;
    
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
__device__ __forceinline__ T clamp_inf_for_half(const float input)
{
    return input;
}

template<>
__device__ __forceinline__ half clamp_inf_for_half(const float input)
{
    return input > 0.0f ? __float2half(min(input, HALF_FLT_MAX - 1000)) : __float2half(max(input, -HALF_FLT_MAX + 1000));
}


