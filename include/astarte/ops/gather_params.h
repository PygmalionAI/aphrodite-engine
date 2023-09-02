#ifndef _ASTARTE_GATHER_PARAMS_H
#define _ASTARTE_GATHER_PARAMS_H

#include "astarte/parallel_tensor.h"
#include "astarte/catype.h"

namespace astarte {

struct GatherParams {
  int legion_dim;
  LayerID layer_guid;
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const;
};

bool operator==(GatherParams const &, GatherParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::GatherParams> {
  size_t operator()(astarte::GatherParams const &) const;
};
} // namespace std

#endif // _ASTARTE_GATHER_PARAMS_H
