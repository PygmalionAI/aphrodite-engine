#ifndef _ASTARTE_SPLIT_PARAMS_H
#define _ASTARTE_SPLIT_PARAMS_H

#include "astarte/parallel_tensor.h"

namespace astarte {

struct SplitParams {
  std::vector<int> splits;
  int legion_axis;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(SplitParams const &, SplitParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::SplitParams> {
  size_t operator()(astarte::SplitParams const &) const;
};
} // namespace std

#endif // _ASTARTE_SPLIT_PARAMS_H
