#ifndef _ASTARTE_GROUPBY_PARAMS_H
#define _ASTARTE_GROUPBY_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct Group_byParams {
  int n;
  float alpha;
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};
bool operator==(Group_byParams const &, Group_byParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::Group_byParams> {
  size_t operator()(astarte::Group_byParams const &) const;
};
} // namespace std

#endif // _ASTARTE_GROUPBY_PARAMS_H
