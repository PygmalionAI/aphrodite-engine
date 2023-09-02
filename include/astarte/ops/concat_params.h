#ifndef _ASTARTE_CONCAT_PARAMS_H
#define _ASTARTE_CONCAT_PARAMS_H

#include "astarte/parallel_tensor.h"

namespace astarte {

struct ConcatParams {
  int axis;

  bool is_valid(std::vector<ParallelTensorShape> const &) const;
};

bool operator==(ConcatParams const &, ConcatParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ConcatParams> {
  size_t operator()(astarte::ConcatParams const &) const;
};
} // namespace std

#endif // _ASTARTE_CONCAT_PARAMS_H