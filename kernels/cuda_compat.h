#pragma once

#ifndef USE_ROCM
  #define APHRODITE_LDG(arg) __ldg(arg)
#else
  #define APHRODITE_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define APHRODITE_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
  #define APHRODITE_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#ifndef USE_ROCM
  #define APHRODITE_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane);
#else
  #define APHRODITE_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif