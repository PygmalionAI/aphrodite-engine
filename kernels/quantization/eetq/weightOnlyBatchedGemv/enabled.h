/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "cutlass_extensions/include/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"
#include "common.h"
#include <cuda_runtime.h>


inline int getSMVersion()
{
    int device{-1};
    cudaGetDevice(&device);
    int sm_major = 0;
    int sm_minor = 0;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    return sm_major * 10 + sm_minor;
}

namespace tensorrt_llm
{
namespace kernels
{
template <typename TypeB, typename Layout>
struct SupportedLayout
{
    static constexpr bool value = false;
};

template <>
struct SupportedLayout<uint8_t, cutlass::layout::ColumnMajorTileInterleave<64, 2>>
{
    static constexpr bool value = true;
};

template <>
struct SupportedLayout<cutlass::uint4b_t, cutlass::layout::ColumnMajorTileInterleave<64, 4>>
{
    static constexpr bool value = true;
};

template <typename TypeB, typename Arch>
bool isEnabled()
{
    using Layout = typename cutlass::gemm::kernel::LayoutDetailsB<TypeB, Arch>::Layout;
    return SupportedLayout<TypeB, Layout>::value;
}

template <typename TypeB>
bool isEnabledForArch(int arch)
{
    if (arch >= 70 && arch < 75)
    {
        return isEnabled<TypeB, cutlass::arch::Sm70>();
    }
    else if (arch >= 75 && arch < 80)
    {
        return isEnabled<TypeB, cutlass::arch::Sm75>();
    }
    else if (arch >= 80 && arch <= 90)
    {
        return isEnabled<TypeB, cutlass::arch::Sm80>();
    }
    else
    {
        // TLLM_CHECK_WITH_INFO(false, "Unsupported Arch");
        assert(0);
        return false;
    }
}

inline bool isWeightOnlyBatchedGemvEnabled(WeightOnlyQuantType qtype)
{
    const int arch = getSMVersion();
    if (qtype == WeightOnlyQuantType::Int4b)
    {
        return isEnabledForArch<cutlass::uint4b_t>(arch);
    }
    else if (qtype == WeightOnlyQuantType::Int8b)
    {
        return isEnabledForArch<uint8_t>(arch);
    }
    else
    {
        assert(0);
        // TLLM_CHECK_WITH_INFO(false, "Unsupported WeightOnlyQuantType");
        return false;
    }
}
} // namespace kernels
} // namespace tensorrt_llm
