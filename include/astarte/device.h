#ifndef _ASTARTE_DEVICE_H
#define _ASTARTE_DEVICE_H

#if defined(CA_USE_CUDA) || defined(CA_USE_HIP_CUDA)
#include <cuda_runtime.h>
#include <cudnn.h>
#elif defined(CA_USE_HIP_ROCM)
#include <hidapi/hip_runtime.h>
#include <miopen/miopen.h>
#else
#error "Unknown device"
#endif

namespace astarte {
#if defined(CA_USE_CUDA) || defined(CA_USE_HIP_CUDA)
typedef cudaStream_t caStream_t;        // codex astarte
cudaError_t get_legion_stream(cudaStream_t *stream);
typedef cudnnTensorDescriptor_t caTensorDescriptor_t;
typedef cudnnActivationDescriptor_t caActivationDescriptor_t;
typedef cudnnPoolingDescriptor_t caPoolingDescriptor_t;
#elif defined(CA_USE_HIP_ROCM)
typedef hipStream_t caStream_t;
hipError_t get_legion_stream(hipStream_t *stream);
typedef miopenTensorDescriptor_t caTensorDescriptor_t;
typedef miopenActivationDescriptor_t caActivationDescriptor_t;
typedef miopenPoolingDescriptor_t caPoolingDescriptor_t;
#else
#error "Unknown device"
#endif
}; // namespace astarte

#endif // _ASTARTE_DEVICE_H