#ifndef _ASTARTE_ARG_TOPK_PARAMS_H
#define _ASTARTE_ARG_TOPK_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ArgTopKParams {
    LayerID layer_guid;
    int k;
    float sorted;
    bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ArgTopKParams const &, ArgTopKParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ArgTopKParams> {
    size_t operator()(astarte::ArgTopKParams const &) const;
};
} // namespace std
#endif // _ASTARTE_ARG_TOPK_PARAMS_H