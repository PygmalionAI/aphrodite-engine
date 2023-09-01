#ifndef _ASTARTE_AGGREGATE_PARAMS_H
#define _ASTARTE_AGGREGATE_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct AggregateParams {
    int n;
    float lambda_bal;
    bool is_valid(std::vector<ParallelTensorShape> const &) const;
};
bool operator==(AggregateParams const &, AggregateParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::AggregateParams> {
    size_t operator()(astarte::AggregateParams const &) const;
};
} // namespace std
#endif // _ASTARTE_AGGREGATE_PARAMS_H