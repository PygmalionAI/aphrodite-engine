#ifndef _ASTARTE_POOL_2D_PARAMS_H
#define _ASTARTE_POOL_2D_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct Pool2DParams {
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims) const;

private:
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const;
};

bool operator==(Pool2DParams const &, Pool2DParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::Pool2DParams> {
  size_t operator()(astarte::Pool2DParams const &) const;
};
} // namespace std

#endif // _ASTARTE_POOL_2D_PARAMS_H
