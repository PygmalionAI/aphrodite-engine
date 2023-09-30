#ifndef _ASTARTE_FUSED_PARALLEL_OP_PARAMS_H
#define _ASTARTE_FUSED_PARALLEL_OP_PARAMS_H

#include "parallel_op_info.h"

namespace astarte {

struct FusedParallelOpParams {
    std::vector<ParallelOpInfo> parallel_ops;
    bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(FusedParallelOpParams const &, FusedParallelOpParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::FusedParallelOpParams> {
    size_t operator()(astarte::FusedParallelOpParams const &) const;
};
} // namespace std

#endif // _ASTARTE_FUSED_PARALLEL_OP_PARAMS_H