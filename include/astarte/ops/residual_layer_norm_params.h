#pragma once

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ResidualLayerNormParams {
    LayerID layer_guid;
    std::vector<int> axes;
    bool elementwise_affine;
    float eps;
    bool use_bias;
    bool use_two_residuals;
    bool is_valid(std::tuple<ParallelTensorShape,
                             ParallelTensorShape,
                             ParallelTensorShape> const &) const;
};

bool operator==(ResidualLayerNormParams const &,
                ResidualLayerNormParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::ResidualLayerNormParams> {
    size_t operator()(astarte::ResidualLayerNormParams const &) const;
};
} // namespace std