#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#define CUDACHECK(cmd)                                               \
  do {                                                               \
    cudaError_t e = cmd;                                             \
    if (e != cudaSuccess) {                                          \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,  \
             cudaGetErrorString(e));                                 \
      exit(EXIT_FAILURE);                                            \ 
    }                                                                \
  } while (0)

namespace aphrodite {
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } start;
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } end;
};

struct MetaData {
  alginas(128) Signal sg;
  alignas(128) int counter;
};
static_assert(offsetof(MetaData, counter) == 128);
static_assert(sizeof(MetaData) == 256);

struct __align__(16) RankData { const void *__restrict__ ptrs[8]; };

struct RankSignals {
  volatile Signal *signals[8];
};


// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(float)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with pytorch, the + operator
// for half and bfloat16 is disabled so we call the instrinsics
// directly
DINLINE half &assign_add(half &a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float &assign_add(float &a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float2(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat162(val);
}
DINLINE nv_bfloat16 &assign_add(nv_bfloat16 &a, nv_bfloat16 b) {
  a = __hadd2(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N> &packed_assign_add(array_t<T, N> &a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; ++i) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

// compute flag at compile time
__host__ __device__ constexpr uint64_t compute_flag(int ngpus) {
  auto m = std::numeric_limits<uint64_t>::max();
  return m >> ((8 - ngpus) * 8);
}

template <int ngpus>
__device__ __forceinline__ void start_sync(const RankSignals &sg,
                                           volatile MetaData *meta, int rank) {
  constexpr auto FLAG = compute_flag(ngpus);
  if (blockIdx.x == 0) {
    if (threadIdx.x < ngpus)
      // simultaneously write to the corresponding byte to all other ranks.
      // Latency = 1 p2p write
      sg.signals[threadIdx.x]->start.data[rank] = 255;
    else if (threadIdx.x == 32)
      // reset
      meta->sg.end.flag = 0;
  }
  if (threadIdx.x == 0) {
    while (meta->sg.start.flag != FLAG)
      ;
  }
  __syncthreads();
}

template <int ngpus, bool final_sync = false>
__device__ __forceinline__ void end_sync(const RankSignals &sg,
                                         volatile MetaData *meta, int rank) {
  constexpr auto FLAG = compute_flag(ngpus);
  __syncthreads();
  __shared__ int num;
  if (threadIdx.x == 0) num = atomicAdd((int *)&meta->counter, 1);
  __syncthreads();

  // Only the last completing block can perform the end sync
  // This can ensure when the final busy ends, all ranks must
  // have finished reading each other's buffer.
  if (num == gridDim.x - 1) {
    if (threadIdx.x == 32) {
      // reset in a different warp
      meta->counter = 0;
      meta->sg.start.flag = 0;
    } else if (threadIdx.x < ngpus) {
      // simultaneously write to the corresponding byte to all other ranks.
      // Latency = 1 p2p write
      sg.signals[threadIdx.x]->end.data[rank] = 255;
    }
    // if this is the final sync, only one block needs it
    // because kernel exit can serve as sync
    if constexpr (final_sync) {
      if (threadIdx.x == 0) {
        while (meta->sg.end.flag != FLAG)
          ;
      }
    }
  }
  if constexpr (!final_sync) {
    if (threadIdx.x == 0) {
      while (meta->sg.end.flag != FLAG)
        ;
    }
    __syncthreads();
  }
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P *ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData *_dp, RankSignals sg,
                               volatile MetaData *meta, T *__restrict__ result,
                               int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const P *ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (P *)_dp->ptrs[target];
  }
  start_sync<ngpus>(sg, meta, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P *)result)[idx] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  end_sync<ngpus, true>(sg, meta, rank);
}

template <typename P>
DINLINE P *get_tmp_buf(volatile Signal *sg) {
  return (P *)(((MetaData *)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData *_dp, RankSignals sg,
                               volatile MetaData *meta, T *__restrict__ result,
                               int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus -1 ? size : start + part;
  const P *ptrs[ngpus];
  P *tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P *)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  start_sync<ngpus>(sg, meta, rank);
  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  // TODO: replace this with per-block release-acquire
  // can save about 1-2 us (not a lot though)
  end_sync<ngpus>(sg, meta, rank);

  // stage 2: reduce
  for (int idx = tid; idx < part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int dst_idx = ((rank + i) % ngpus) * part + idx;
      ((P *)result)[dst_idx] = tmps[i][idx];
    }
  }
  // process the last larger partition
  int remaining = size - part * ngpus;
  if (tid < remaining) {
    int dst_idx = tid + part * ngpus;
    ((P *)result)[dst_idx] = get_tmp_buf<P>(sg.signals[ngpus - 1])[part + tid];
  }

}

