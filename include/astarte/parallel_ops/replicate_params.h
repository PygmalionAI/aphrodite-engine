#ifndef _ASTARTE_REPLICATE_PARAMS_H
#define _ASTARTE_REPLICATE_PARAMS_H

namespace astarte {

struct ReplicateParams {
    int replicate_legion_dim;
    int replicate_degree;
    bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(ReplicateParams const &, ReplicateParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ReplicateParams> {
    size_t operator()(astarte::ReplicateParams const &) const;
};
} // namespace std

#endif // _ASTARTE_REPLICATE_PARAMS_H