#include <cuda.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include "qgemm_kernel.hpp"


template <
  typename SMs,
  typename T,
  typename TQ,
  typename T2,
  typename NumBits,
  typename GroupSize
>
void
_qgemm_raw(int M,
           int N,
           int K,
           int P,
           const T * const __restrict__ A,
           const TQ* const __restrict__ Q,
                 T *       __restrict__ D,
           const T * const __restrict__ S,
           const T * const __restrict__ QM,
           const T2* const __restrict__ QM2,
               void*       __restrict__ workspace,
           const int                    template_id,
           const cudaStream_t           stream)
{

    using namespace cute;
    static constexpr config::QuantMapModeEnum      kVectorized    = config::QuantMapModeEnum     ::Vectorized;
    static constexpr config::QuantMapModeEnum      kVectorized_32 = config::QuantMapModeEnum     ::Vectorized_32;
    static constexpr config::QuantMapModeEnum      kVectorized_16 = config::QuantMapModeEnum     ::Vectorized_16;
    static constexpr config::QuantMapModeEnum      kVectorized_8  = config::QuantMapModeEnum     ::Vectorized_8;
    static constexpr config::QuantMapModeEnum      kMarlin        = config::QuantMapModeEnum     ::Marlin;
    static constexpr config::AccumulationModeEnum  kLow           = config::AccumulationModeEnum ::Low;
    static constexpr config::AccumulationModeEnum  kHigh          = config::AccumulationModeEnum ::High;
    static constexpr config::AccumulationModeEnum  kMixed         = config::AccumulationModeEnum ::Mixed;
    static constexpr config::DecompositionModeEnum kStreamK       = config::DecompositionModeEnum::StreamK;

#define RUN_QGEMM(T,                           \
                  TQ,                          \
                  T2,                          \
                  SLICES,                      \
                  BLOCKS,                      \
                  THREADS,                     \
                  TILE_M,                      \
                  TILE_K,                      \
                  TILE_P,                      \
                  STAGES,                      \
                  NUM_BITS,                    \
                  GROUP_SIZE,                  \
                  QUANT_MAP_MODE,              \
                  ACCUMULATION_MODE,           \
                  DECOMPOSITION_MODE,          \
                  G2S_TILED_COPY_SIZE_S,       \
                  MMA_PRM_K)                   \
    do {                                       \
        qgemm_host<                            \
            T,                                 \
            TQ,                                \
            T2,                                \
            cute::Int<SLICES>,                 \
            cute::Int<BLOCKS>,                 \
            cute::Int<THREADS>,                \
            cute::Int<TILE_M>,                 \
            cute::Int<TILE_K>,                 \
            cute::Int<TILE_P>,                 \
            cute::Int<STAGES>,                 \
            cute::Int<NUM_BITS>,               \
            cute::Int<GROUP_SIZE>,             \
            QUANT_MAP_MODE,                    \
            ACCUMULATION_MODE,                 \
            DECOMPOSITION_MODE,                \
            cute::Int<G2S_TILED_COPY_SIZE_S>,  \
            cute::Int<MMA_PRM_K>               \
        > (                                    \
            M,                                 \
            N,                                 \
            K,                                 \
            P,                                 \
            A,                                 \
            Q,                                 \
            D,                                 \
            S,                                 \
            QM,                                \
            QM2,                               \
            workspace,                         \
            stream);                           \
    } while (false)

    // Generated Code Below
}


#define INSTANTIATE_TEMPLATE(SMS,                  \
                             T,                    \
                             TQ,                   \
                             T2,                   \
                             NUM_BITS,             \
                             GROUP_SIZE)           \
    template                                       \
    void                                           \
    _qgemm_raw<                                    \
        cute::Int<SMS>,                            \
        T,                                         \
        TQ,                                        \
        T2,                                        \
        cute::Int<NUM_BITS>,                       \
        cute::Int<GROUP_SIZE>                      \
    > (                                            \
        int M,                                     \
        int N,                                     \
        int K,                                     \
        int P,                                     \
        const T * const __restrict__ A,            \
        const TQ* const __restrict__ Q,            \
              T *       __restrict__ D,            \
        const T * const __restrict__ S,            \
        const T * const __restrict__ QM,           \
        const T2* const __restrict__ QM2,          \
            void*       __restrict__ workspace,    \
        const int                    template_id,  \
        const cudaStream_t           stream)


INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 2, 32);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 2, 64);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 2, 128);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 2, 256);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 3, 32);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 3, 64);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 3, 128);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 3, 256);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 4, 32);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 4, 64);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 4, 128);
INSTANTIATE_TEMPLATE(84 , cute::half_t    , cute::uint16_t, __half2       , 4, 256);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 2, 32);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 2, 64);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 2, 128);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 2, 256);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 3, 32);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 3, 64);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 3, 128);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 3, 256);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 4, 32);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 4, 64);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 4, 128);
INSTANTIATE_TEMPLATE(108, cute::half_t    , cute::uint16_t, __half2       , 4, 256);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 2, 32);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 2, 64);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 2, 128);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 2, 256);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 3, 32);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 3, 64);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 3, 128);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 3, 256);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 4, 32);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 4, 64);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 4, 128);
INSTANTIATE_TEMPLATE(128, cute::half_t    , cute::uint16_t, __half2       , 4, 256);

INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 32);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 64);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 128);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 256);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 32);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 64);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 128);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 256);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 32);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 64);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 128);
INSTANTIATE_TEMPLATE(84 , cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 256);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 32);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 64);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 128);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 256);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 32);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 64);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 128);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 256);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 32);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 64);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 128);
INSTANTIATE_TEMPLATE(108, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 256);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 32);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 64);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 128);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 2, 256);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 32);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 64);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 128);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 3, 256);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 32);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 64);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 128);
INSTANTIATE_TEMPLATE(128, cute::bfloat16_t, cute::uint16_t, __nv_bfloat162, 4, 256);
