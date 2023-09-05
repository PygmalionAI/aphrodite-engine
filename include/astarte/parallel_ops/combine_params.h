#ifndef _ASTARTE_COMBINE_PARAMS_H
#define _ASTARTE_COMBINE_PARAMS_H
#include "astarte/parallel_tensor.h"

namespace astarte {

struct CombineParams {
    int combine_legion_dim;
    int combine_degree;
    bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(CombineParams const &, CombineParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::CombineParams> {
    size_t operator()(astarte::CombineParams const &) const;
};
} // namespace std
#endif // _ASTARTE_COMBINE_PARAMS_H
