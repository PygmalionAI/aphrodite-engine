#ifndef _ASTARTE_REDUCTION_PARAMS_H
#define _ASTARTE_REDUCTION_PARAMS_H

namespace astarte {

struct ReductionParams {
    int reduction_legion_dim;
    int reduction_degree;
    bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(ReductionParams const &, ReductionParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ReductionParams> {
    size_t operator()(astarte::ReductionParams const &) const;
};
} // namespace std

#endif // _ASTARTE_REDUCTION_PARAMS_H