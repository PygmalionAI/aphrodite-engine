#ifndef _ASTARTE_CAST_PARAMS_H
#define _ASTARTE_CAST_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct CastParams {
  DataType dtype;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(CastParams const &, CastParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::CastParams> {
  size_t operator()(astarte::CastParams const &) const;
};
} // namespace std

#endif // _ASTARTE_CAST_PARAMS_H