#ifndef _ASTARTE_SOFTMAX_PARAMS_H
#define _ASTARTE_SOFTMAX_PARAMS_H

#include "astarte/parallel_tensor.h"

namespace astarte {

struct SoftmaxParams {
  int dim;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(SoftmaxParams const &, SoftmaxParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::SoftmaxParams> {
  size_t operator()(astarte::SoftmaxParams const &) const;
};
} // namespace std

#endif // _ASTARTE_SOFTMAX_PARAMS_H
