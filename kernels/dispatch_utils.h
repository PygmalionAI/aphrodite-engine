/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */
#include <torch/extension.h>

#define APHRODITE_DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)   \
  // AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)

#define APHRODITE_DISPATCH_CASE_QUANT_TYPES(...)              \
  APHRODITE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)         \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)


#define APHRODITE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
#define APHRODITE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, APHRODITE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
    TYPE, NAME, APHRODITE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define APHRODITE_DISPATCH_QUANT_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, APHRODITE_DISPATCH_CASE_QUANT_TYPES(__VA_ARGS__))