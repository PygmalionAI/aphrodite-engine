#ifndef _ASTARTE_BEAM_TOPK_PARAMS_H
#define _ASTARTE_BEAM_TOPK_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct BeamTopKParams {
  LayerID layer_guid;
  bool sorted;
  int max_beam_width;
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(BeamTopKParams const &, BeamTopKParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::BeamTopKParams> {
  size_t operator()(astarte::BeamTopKParams const &) const;
};
} // namespace std

#endif // _ASTARTE_BEAM_TOPK_PARAMS_H