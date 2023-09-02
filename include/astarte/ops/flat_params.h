#ifndef _ASTARTE_FLAT_PARAMS_H
#define _ASTARTE_FLAT_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct FlatParams {
  bool is_valid(ParallelTensorShape const &) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims) const;

private:
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const;
};

bool operator==(FlatParams const &, FlatParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::FlatParams> {
  size_t operator()(astarte::FlatParams const &) const;
};
} // namespace std

#endif // _ASTARTE_FLAT_PARAMS_H
