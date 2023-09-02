#pragma once

#include "astarte/parallel_tensor.h"

namespace astarte {

struct TransposeParams {
  std::vector<int> perm;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(TransposeParams const &, TransposeParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::TransposeParams> {
  size_t operator()(astarte::TransposeParams const &) const;
};
} // namespace std
