#pragma once

#include "astarte/parallel_tensor.h"

namespace astarte {

struct BatchMatmulParams {
  int a_seq_length_dim, b_seq_length_dim;
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(BatchMatmulParams const &, BatchMatmulParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::BatchMatmulParams> {
  size_t operator()(astarte::BatchMatmulParams const &) const;
};
} // namespace std