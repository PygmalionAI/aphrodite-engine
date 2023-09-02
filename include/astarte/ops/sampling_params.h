#ifndef _ASTARTE_SAMPLING_PARAMS_H
#define _ASTARTE_SAMPLING_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct SamplingParams {
  float top_p;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(SamplingParams const &, SamplingParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::SamplingParams> {
  size_t operator()(astarte::SamplingParams const &) const;
};
} // namespace std

#endif // _ASTARTE_SAMPLING_PARAMS_H