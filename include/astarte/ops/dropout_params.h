#ifndef _ASTARTE_DROPOUT_PARAMS_H
#define _ASTARTE_DROPOUT_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct DropoutParams {
  float rate;
  unsigned long long seed;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(DropoutParams const &, DropoutParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::DropoutParams> {
  size_t operator()(astarte::DropoutParams const &) const;
};
} // namespace std

#endif // _ASTARTE_DROPOUT_PARAMS_H