/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "causal_conv1d.h"

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)          \
  if (ITYPE == at::ScalarType::Half) {                                    \
    using input_t = at::Half;                                             \
    __VA_ARGS__();                                                        \
  } else if (ITYPE == at::ScalarType::BFloat16) {                         \
    using input_t = at::BFloat16;                                         \
    __VA_ARGS__();                                                        \
  } else if (ITYPE == at::ScalarType::Float) {                            \
    using input_t = float;                                                \
    __VA_ARGS__();                                                        \
  } else {                                                                \
    AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), \
             "'");                                                        \
  }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)           \
  if (WTYPE == at::ScalarType::Half) {                                     \
    using weight_t = at::Half;                                             \
    __VA_ARGS__();                                                         \
  } else if (WTYPE == at::ScalarType::BFloat16) {                          \
    using weight_t = at::BFloat16;                                         \
    __VA_ARGS__();                                                         \
  } else if (WTYPE == at::ScalarType::Float) {                             \
    using weight_t = float;                                                \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), \
             "'");                                                         \
  }

template <typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase& params, cudaStream_t stream);
template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase& params,
                                        cudaStream_t stream);

template <typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase& params, cudaStream_t stream);

void set_conv_params_fwd(ConvParamsBase& params,
                         // sizes
                         const size_t batch, const size_t dim,
                         const size_t seqlen, const size_t width,
                         // device pointers
                         const at::Tensor x, const at::Tensor weight,
                         const at::Tensor out, void* bias_ptr,
                         bool silu_activation) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.width = width;

  params.silu_activation = silu_activation;

  // Set the pointers and strides.
  params.x_ptr = x.data_ptr();
  params.weight_ptr = weight.data_ptr();
  params.bias_ptr = bias_ptr;
  params.out_ptr = out.data_ptr();
  // All stride are in elements, not bytes.
  params.x_batch_stride = x.stride(0);
  params.x_c_stride = x.stride(1);
  params.x_l_stride = x.stride(-1);
  params.weight_c_stride = weight.stride(0);
  params.weight_width_stride = weight.stride(1);
  params.out_batch_stride = out.stride(0);
  params.out_c_stride = out.stride(1);
  params.out_l_stride = out.stride(-1);
}

at::Tensor causal_conv1d_fwd(const at::Tensor& x, const at::Tensor& weight,
                             const c10::optional<at::Tensor>& bias_,
                             const c10::optional<at::Tensor>& seq_idx_,
                             const c10::optional<at::Tensor>& initial_states_,
                             c10::optional<at::Tensor>& final_states_out_,
                             bool silu_activation) {
  auto input_type = x.scalar_type();
  auto weight_type = weight.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::Float ||
              input_type == at::ScalarType::Half ||
              input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(weight_type == at::ScalarType::Float ||
              weight_type == at::ScalarType::Half ||
              weight_type == at::ScalarType::BFloat16);

  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(weight.is_cuda());

  const auto sizes = x.sizes();
  const int batch_size = sizes[0];
  const int dim = sizes[1];
  const int seqlen = sizes[2];
  const int width = weight.size(-1);

  CHECK_SHAPE(x, batch_size, dim, seqlen);
  CHECK_SHAPE(weight, dim, width);

  TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
  const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;

  if (is_channel_last) {
    TORCH_CHECK(
        dim % 8 == 0,
        "causal_conv1d only supports channel dimension divisible by 8 for now");
    TORCH_CHECK(x.stride(2) % 8 == 0 and x.stride(0) % 8 == 0,
                "causal_conv1d with channel last layout requires strides "
                "(x.stride(0) and x.stride(2)) to be multiples of 8");
  }
  TORCH_CHECK(width >= 2 && width <= 4,
              "causal_conv1d only supports width between 2 and 4");

  if (bias_.has_value()) {
    auto bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type);
    TORCH_CHECK(bias.is_cuda());
    TORCH_CHECK(bias.stride(-1) == 1);
    CHECK_SHAPE(bias, dim);
  }

  if (seq_idx_.has_value()) {
    TORCH_CHECK(is_channel_last,
                "seq_idx is only supported for channel last layout");
    auto seq_idx = seq_idx_.value();
    TORCH_CHECK(seq_idx.scalar_type() == torch::kInt32);
    TORCH_CHECK(seq_idx.is_cuda());
    TORCH_CHECK(seq_idx.is_contiguous());
    CHECK_SHAPE(seq_idx, batch_size, seqlen);
  }

  at::Tensor out = torch::empty_like(x);

  ConvParamsBase params;
  set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                      bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                      silu_activation);

  if (seq_idx_.has_value()) {
    params.seq_idx_ptr = seq_idx_.value().data_ptr();
  } else {
    params.seq_idx_ptr = nullptr;
  }

  if (initial_states_.has_value()) {
    TORCH_CHECK(is_channel_last,
                "initial_states is only supported for channel last layout");
    auto initial_states = initial_states_.value();
    TORCH_CHECK(initial_states.scalar_type() == input_type);
    TORCH_CHECK(initial_states.is_cuda());
    CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
    TORCH_CHECK(initial_states.stride(1) == 1);
    params.initial_states_ptr = initial_states.data_ptr();
    params.initial_states_batch_stride = initial_states.stride(0);
    params.initial_states_c_stride = initial_states.stride(1);
    params.initial_states_l_stride = initial_states.stride(2);
  } else {
    params.initial_states_ptr = nullptr;
  }

  if (final_states_out_.has_value()) {
    TORCH_CHECK(is_channel_last,
                "final_states is only supported for channel last layout");
    auto final_states = final_states_out_.value();
    TORCH_CHECK(final_states.scalar_type() == input_type);
    TORCH_CHECK(final_states.is_cuda());
    CHECK_SHAPE(final_states, batch_size, dim, width - 1);
    TORCH_CHECK(final_states.stride(1) == 1);
    params.final_states_ptr = final_states.data_ptr();
    params.final_states_batch_stride = final_states.stride(0);
    params.final_states_c_stride = final_states.stride(1);
    params.final_states_l_stride = final_states.stride(2);
  } else {
    params.final_states_ptr = nullptr;
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(
      x.scalar_type(), "causal_conv1d_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(
            weight.scalar_type(), "causal_conv1d_fwd", [&] {
              if (!is_channel_last) {
                causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
              } else {
                causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params,
                                                                      stream);
              }
            });
      });
  return out;
}

at::Tensor causal_conv1d_update(const at::Tensor& x,
                                const at::Tensor& conv_state,
                                const at::Tensor& weight,
                                const c10::optional<at::Tensor>& bias_,
                                bool silu_activation) {
  auto input_type = x.scalar_type();
  auto weight_type = weight.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::Float ||
              input_type == at::ScalarType::Half ||
              input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(weight_type == at::ScalarType::Float ||
              weight_type == at::ScalarType::Half ||
              weight_type == at::ScalarType::BFloat16);
  TORCH_CHECK(conv_state.scalar_type() == input_type);

  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(conv_state.is_cuda());
  TORCH_CHECK(weight.is_cuda());

  const auto sizes = x.sizes();
  const int batch_size = sizes[0];
  const int dim = sizes[1];
  const int width = weight.size(-1);

  CHECK_SHAPE(x, batch_size, dim);
  CHECK_SHAPE(conv_state, batch_size, dim, width);
  CHECK_SHAPE(weight, dim, width);

  TORCH_CHECK(width >= 2 && width <= 4,
              "causal_conv1d only supports width between 2 and 4");

  if (bias_.has_value()) {
    auto bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type);
    TORCH_CHECK(bias.is_cuda());
    TORCH_CHECK(bias.stride(-1) == 1);
    CHECK_SHAPE(bias, dim);
  }

  at::Tensor out = torch::empty_like(x);

  ConvParamsBase params;
  set_conv_params_fwd(
      params, batch_size, dim, /*seqlen=*/1, width, x, weight, out,
      bias_.has_value() ? bias_.value().data_ptr() : nullptr, silu_activation);
  params.conv_state_ptr = conv_state.data_ptr();
  // All stride are in elements, not bytes.
  params.conv_state_batch_stride = conv_state.stride(0);
  params.conv_state_c_stride = conv_state.stride(1);
  params.conv_state_l_stride = conv_state.stride(2);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(
      x.scalar_type(), "causal_conv1d_update", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(
            weight.scalar_type(), "causal_conv1d_update", [&] {
              causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
            });
      });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_conv1d_fwd", &causal_conv1d_fwd, "Causal conv1d forward");
  m.def("causal_conv1d_update", &causal_conv1d_update, "Causal conv1d update");
}