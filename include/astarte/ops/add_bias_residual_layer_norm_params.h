#pragma once

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct AddBiasResidualLayerNormParams {
    LayerID layer_guid;
    std::vector<int> axes;
    bool elementwise_affine;
    float eps;
    bool use_bias;
    bool is_valid(
        std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(AddBiasResidualLayerNormParams const &,
                AddBiasResidualLayerNormParams const &);
} // namespace astarte

namespace std {
template <>
struct hash<astarte::AddBiasResidualLayerNormParams> {
    size_t operator()(astarte::AddBiasResidualLayerNormParams const &) const;
};
} // namespace std