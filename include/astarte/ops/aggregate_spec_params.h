#ifndef _ASTARTE_AGGREGATE_SPEC_PARAMS_H
#define _ASTARTE_AGGREGATE_SPEC_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct AggregateSpecParams {
    int n;
    float lambda_bal;
    bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(AggregateSpecParams const &, AggregateSpecParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::AggregateSpecParams> {
    size_t operator()(astarte::AggregateSpecParams const &) const;
};
} // namespace std
#endif // _ASTARTE_AGGREGATE_SPEC_PARAMS_H