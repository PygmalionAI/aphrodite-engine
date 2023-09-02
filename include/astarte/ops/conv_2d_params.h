#ifndef _ASTARTE_CONV_2D_PARAMS_H
#define _ASTARTE_CONV_2D_PARAMS_H

#include "astarte/caconst.h"
#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct Conv2DParams {
  LayerID layer_guid;
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  ActiMode activation;
  bool use_bias;

  bool is_valid(ParallelTensorShape const &input) const;
  void solve_dims(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM],
                  int *output_ndims,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM],
                  int *kernel_ndims,
                  ParallelDim bias_dims[MAX_TENSOR_DIM],
                  int *bias_ndims) const;

  friend bool operator==(Conv2DParams const &lhs, Conv2DParams const &rhs);

private:
  void mark_replica_dims(ParallelTensorShape const &input,
                         ParallelDim output_dims[MAX_TENSOR_DIM],
                         ParallelDim kernel_dims[MAX_TENSOR_DIM],
                         ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
  int output_size(ParallelTensorShape const &input,
                  ParallelDim output_dims[MAX_TENSOR_DIM]) const;
  int kernel_size(ParallelTensorShape const &input_shape,
                  ParallelDim kernel_dims[MAX_TENSOR_DIM]) const;
  int bias_size(ParallelTensorShape const &input,
                ParallelDim bias_dims[MAX_TENSOR_DIM]) const;
};

} // namespace astarte

namespace std {
template <>
struct hash<astarte::Conv2DParams> {
  size_t operator()(astarte::Conv2DParams const &) const;
};
} // namespace std

#endif // _ASTARTE_CONV_2D_PARAMS_H