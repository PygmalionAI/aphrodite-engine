#ifndef _ASTARTE_ARGMAX_PARAMS_H
#define _ASTARTE_ARGMAX_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ArgMaxParams {
    bool beam_search;
    bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ArgMaxParams const &, ArgMaxParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ArgMaxParams> {
    size_t operator()(astarte::ArgMaxParams const &) const;
};
} // namespace std
#endif // _ASTARTE_ARGMAX_PARAMS_H