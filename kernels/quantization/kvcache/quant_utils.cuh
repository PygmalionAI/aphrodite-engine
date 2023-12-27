#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>

#include "../../attention/attention_dtypes.h"
#include "../../attention/dtype_float32.cuh"
#include "../../attention/dtype_float16.cuh"
#include "../../attention/dtype_bfloat16.cuh"

using namespace aphrodite;

template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}

// fp8 -> half
template<>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(const uint8_t& a)
{
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);
    return res.x;
}

// fp8x2 -> half2
template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(const uint16_t& a)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;
    __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E5M2);
    tmp.u16[0] = res.x;
    tmp.u16[1] = res.y;
    return tmp.u32;
}

// fp8x4 -> half4
template<>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t& a)
{
    union {
        uint2 u32x2;
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a);
    tmp.u32[1] = vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U));
    return tmp.u32x2;
}

// fp8x8 -> half8
template<>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2& a)
{
    union {
        uint4 u64x4;
        uint32_t u64[2];
    } tmp;
    tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x);
    tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y);
    return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template<>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a)
{
    // Note there is no direct convert function from fp8 to bfloat16
    // So we convert fp8 to half first, then convert half to bfloat16
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, __NV_E5M2);
    // half -> float -> bfloat16
    float tmp = half_to_float(res.x);
    return __float2bfloat16(tmp);

}

// fp8x2 -> bf16_4_t
template<>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a)
{
    bf16_4_t res;
    // uint16_t hi = (uint16_t)(a >> 16U);
    res.x = vec_conversion<__nv_bfloat16, uint16_t>((uint16_t)a);
    res.y = vec_conversion<__nv_bfloat16, uint16_t>((uint16_t)(a >> 16U));
    return res;
}

// fp8x8 -> bf16_8_t
template<>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint4>(const uint4& a)
{
    bf16_4_t tmp1, tmp2;
    tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x);
    tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y);
    bf16_8_t res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}

// fp8 -> float
template<>
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a)
{
    // fp8 -> half
    uint16_t tmp = vec_conversion<uint16_t, uint8_t>(a);
    // half -> float
    return half_to_float(tmp);
}

