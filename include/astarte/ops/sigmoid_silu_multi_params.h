#pragma once

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct SigmoidSiluMultiParams {
    LayerID layer_guid;
    bool is_valid(
        std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(SigmoidSiluMultiParams const &, SigmoidSiluMultiParams const &);
} // namespace astarte

namespace std {
template <>
struct hash<astarte::SigmoidSiluMultiParams> {
    size_t operator()(astarte::SigmoidSiluMultiParams const &) const;
};
} // namespace std