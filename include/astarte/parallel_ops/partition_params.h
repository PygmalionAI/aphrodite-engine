#ifndef _ASTARTE_PARTITION_PARAMS_H
#define _ASTARTE_PARTITION_PARAMS_H

namespace astarte {

struct RepartitionParams {
    int repartition_legion_dim;
    int repartition_degree;
    bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(RepartitionParams const &, RepartitionParams const &);
} // namespace astarte

namespace std {
template <>
struct hash<astarte::RepartitionParams> {
    size_t operator()(astarte::RepartitionParams const &) const;
};
} // namespace astarte

#endif // _ASTARTE_PARTITION_PARAMS_H