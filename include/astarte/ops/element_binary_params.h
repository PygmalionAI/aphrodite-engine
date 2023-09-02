#ifndef _ASTARTE_ELEMENT_BINARY_PARAMS_H
#define _ASTARTE_ELEMENT_BINARY_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ElementBinaryParams {
  LayerID layer_guid;
  OperatorType type;
  bool inplace_a;

  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(ElementBinaryParams const &, ElementBinaryParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ElementBinaryParams> {
  size_t operator()(astarte::ElementBinaryParams const &) const;
};
} // namespace std

#endif // _ASTARTE_ELEMENT_BINARY_PARAMS_H
