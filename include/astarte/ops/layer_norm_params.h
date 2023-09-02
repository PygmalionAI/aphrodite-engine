#pragma once

#include "astarte/parallel_tensor.h"

namespace astarte {

struct LayerNormParams {
  LayerID layer_guid;
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(LayerNormParams const &, LayerNormParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::LayerNormParams> {
  size_t operator()(astarte::LayerNormParams const &) const;
};
} // namespace std
