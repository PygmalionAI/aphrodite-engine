#ifndef _ASTARTE_TOPK_PARAMS_H
#define _ASTARTE_TOPK_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct TopKParams {
  int k;
  bool sorted;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(TopKParams const &, TopKParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::TopKParams> {
  size_t operator()(astarte::TopKParams const &) const;
};
} // namespace std

#endif // _ASTARTE_TOPK_PARAMS_H
