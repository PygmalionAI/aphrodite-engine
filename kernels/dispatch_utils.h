/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */
#include <torch/extension.h>

#define APHRODITE_DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define APHRODITE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, APHRODITE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifdef APHRODITE_BUILD_CPU_ONLY
#define APHRODITE_DISPATCH_TO_CUDA_CASE(BASENAME, ...) 
#else
#define APHRODITE_DISPATCH_TO_CUDA_CASE(BASENAME, ...)                              \
  case c10::DeviceType::CUDA: {                                                \
    return BASENAME(__VA_ARGS__);                                              \
  }
#endif

#ifdef APHRODITE_BUILD_CPU_OPS
#define APHRODITE_DISPATCH_TO_CPU_CASE(BASENAME, ...)                               \
  case c10::DeviceType::CPU: {                                                 \
    return BASENAME##_cpu(__VA_ARGS__);                                        \
  }
#else
#define APHRODITE_DISPATCH_TO_CPU_CASE(BASENAME, ...)
#endif

#define APHRODITE_DISPATCH_DEVICES(DEVICE, BASENAME, ...)                           \
  {                                                                            \
    auto device = DEVICE.type();                                               \
    switch (device) {                                                          \
      APHRODITE_DISPATCH_TO_CUDA_CASE(BASENAME, __VA_ARGS__)                        \
      APHRODITE_DISPATCH_TO_CPU_CASE(BASENAME, __VA_ARGS__)                         \
    default:                                                                   \
      AT_ERROR('"', #BASENAME, "\" not implemented for '",                      \
               c10::DeviceTypeName(device), "'");                              \
    }                                                                          \
  }
