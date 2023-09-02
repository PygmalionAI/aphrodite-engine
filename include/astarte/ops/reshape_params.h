#ifndef _ASTARTE_RESHAPE_PARAMS_H
#define _ASTARTE_RESHAPE_PARAMS_H

#include "astarte/parallel_tensor.h"

namespace astarte {

struct ReshapeParams {
  std::vector<int> shape;
  LayerID layer_guid;

  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ReshapeParams const &, ReshapeParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ReshapeParams> {
  size_t operator()(astarte::ReshapeParams const &) const;
};
} // namespace std

#endif // _ASTARTE_RESHAPE_PARAMS_H
