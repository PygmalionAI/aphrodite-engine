#pragma once

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct ReduceParams {
    std::vector<int> axes;
    bool keepdims;
    LayerID layer_guid;

    bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(ReduceParams const &, ReduceParams const &);
} //namespace astarte

namespace std {
template<>
struct hash<astarte::ReduceParams> {
    size_t operator()(astarte::ReduceParams const &) const;
};
} //namespace std