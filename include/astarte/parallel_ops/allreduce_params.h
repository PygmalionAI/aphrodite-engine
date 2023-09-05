#ifndef _ASTARTE_ALLREDUCE_PARAMS_H
#define _ASTARTE_ALLREDUCE_PARAMS_H
#include "astarte/parallel_tensor.h"

namespace astarte {

struct AllReduceParams {
    int allreduce_legion_dim;
    bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(AllReduceParams const &, AllReduceParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::AllReduceParams> {
    size_t operator()(astarte::AllReduceParams const &) const;
};
} // namespace std

#endif // _ASTARTE_ALLREDUCE_PARAMS_H