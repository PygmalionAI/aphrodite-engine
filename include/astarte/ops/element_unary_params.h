#ifndef _ASTARTE_ELEMENTARY_UNARY_PARAMS_H
#define _ASTARTE_ELEMENTARY_UNARY_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ElementUnaryParams {
  OperatorType op_type;
  bool inplace;
  float scalar = 0.0;
  LayerID layer_guid;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(ElementUnaryParams const &, ElementUnaryParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ElementUnaryParams> {
  size_t operator()(astarte::ElementUnaryParams const &) const;
};
} // namespace std

#endif // _ASTARTE_ELEMENTARY_UNARY_PARAMS_H
