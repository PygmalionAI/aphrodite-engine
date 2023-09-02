#ifndef _ASTARTE_RMSNORM_PARAMS_H
#define _ASTARTE_RMSNORM_PARAMS_H

#include "astarte/parallel_tensor.h"

namespace astarte {

struct RMSNormParams {
  LayerID layer_guid;
  float eps;
  int dim;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(RMSNormParams const &, RMSNormParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::RMSNormParams> {
  size_t operator()(astarte::RMSNormParams const &) const;
};
} // namespace std

#endif // _ASTARTE_RMSNORM_PARAMS_H