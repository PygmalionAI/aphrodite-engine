#ifndef _ASTARTE_RESIDUAL_RMSNORM_PARAMS_H
#define _ASTARTE_RESIDUAL_RMSNORM_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ResidualRMSNormParams {
    LayerID layer_guid;
    float eps;
    int dim;
    bool is_valid(
        std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const;
};

bool operator==(ResidualRMSNormParams const &, ResidualRMSNormParams const &);
} // namespace astarte

namespace std {
template <>
struct hash<astarte::ResidualRMSNormParams> {
    size_t operator()(astarte::ResidualRMSNormParams const &) const;
};
} // namespace std

#endif // _ASTARTE_RESIDUAL_RMSNORM_PARAMS_H