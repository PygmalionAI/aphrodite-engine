/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <vector>

#include "selective_scan.h"

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

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                 \
  if (WTYPE == at::ScalarType::Float) {                                    \
    using weight_t = float;                                                \
    __VA_ARGS__();                                                         \
  } else if (WTYPE == at::ScalarType::ComplexFloat) {                      \
    using weight_t = c10::complex<float>;                                  \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), \
             "'");                                                         \
  }

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase& params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase& params,
                        // sizes
                        const size_t batch, const size_t dim,
                        const size_t seqlen, const size_t dstate,
                        const size_t n_groups, const size_t n_chunks,
                        const bool is_variable_B, const bool is_variable_C,
                        // device pointers
                        const at::Tensor u, const at::Tensor delta,
                        const at::Tensor A, const at::Tensor B,
                        const at::Tensor C, const at::Tensor out,
                        const at::Tensor z, const at::Tensor out_z, void* D_ptr,
                        void* delta_bias_ptr, void* x_ptr, bool has_z,
                        bool delta_softplus) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.dstate = dstate;
  params.n_groups = n_groups;
  params.n_chunks = n_chunks;
  params.dim_ngroups_ratio = dim / n_groups;

  params.delta_softplus = delta_softplus;

  params.is_variable_B = is_variable_B;
  params.is_variable_C = is_variable_C;

  // Set the pointers and strides.
  params.u_ptr = u.data_ptr();
  params.delta_ptr = delta.data_ptr();
  params.A_ptr = A.data_ptr();
  params.B_ptr = B.data_ptr();
  params.C_ptr = C.data_ptr();
  params.D_ptr = D_ptr;
  params.delta_bias_ptr = delta_bias_ptr;
  params.out_ptr = out.data_ptr();
  params.x_ptr = x_ptr;
  params.z_ptr = has_z ? z.data_ptr() : nullptr;
  params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
  // All stride are in elements, not bytes.
  params.A_d_stride = A.stride(0);
  params.A_dstate_stride = A.stride(1);
  if (!is_variable_B) {
    params.B_d_stride = B.stride(0);
  } else {
    params.B_batch_stride = B.stride(0);
    params.B_group_stride = B.stride(1);
  }
  params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
  if (!is_variable_C) {
    params.C_d_stride = C.stride(0);
  } else {
    params.C_batch_stride = C.stride(0);
    params.C_group_stride = C.stride(1);
  }
  params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
  params.u_batch_stride = u.stride(0);
  params.u_d_stride = u.stride(1);
  params.delta_batch_stride = delta.stride(0);
  params.delta_d_stride = delta.stride(1);
  if (has_z) {
    params.z_batch_stride = z.stride(0);
    params.z_d_stride = z.stride(1);
    params.out_z_batch_stride = out_z.stride(0);
    params.out_z_d_stride = out_z.stride(1);
  }
  params.out_batch_stride = out.stride(0);
  params.out_d_stride = out.stride(1);
}

std::vector<at::Tensor> selective_scan_fwd(
    const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D_, const c10::optional<at::Tensor>& z_,
    const c10::optional<at::Tensor>& delta_bias_, bool delta_softplus) {
  auto input_type = u.scalar_type();
  auto weight_type = A.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::Float ||
              input_type == at::ScalarType::Half ||
              input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(weight_type == at::ScalarType::Float ||
              weight_type == at::ScalarType::ComplexFloat);

  const bool is_variable_B = B.dim() >= 3;
  const bool is_variable_C = C.dim() >= 3;
  const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

  TORCH_CHECK(delta.scalar_type() == input_type);
  TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
  TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

  TORCH_CHECK(u.is_cuda());
  TORCH_CHECK(delta.is_cuda());
  TORCH_CHECK(A.is_cuda());
  TORCH_CHECK(B.is_cuda());
  TORCH_CHECK(C.is_cuda());

  TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
  TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

  const auto sizes = u.sizes();
  const int batch_size = sizes[0];
  const int dim = sizes[1];
  const int seqlen = sizes[2];
  const int dstate = A.size(1);
  const int n_groups = is_variable_B ? B.size(1) : 1;

  TORCH_CHECK(dstate <= 256,
              "selective_scan only supports state dimension <= 256");

  CHECK_SHAPE(u, batch_size, dim, seqlen);
  CHECK_SHAPE(delta, batch_size, dim, seqlen);
  CHECK_SHAPE(A, dim, dstate);
  if (!is_variable_B) {
    CHECK_SHAPE(B, dim, dstate);
  } else {
    CHECK_SHAPE(B, batch_size, n_groups, dstate,
                !is_complex ? seqlen : seqlen * 2);
    TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
  }
  if (!is_variable_C) {
    CHECK_SHAPE(C, dim, dstate);
  } else {
    CHECK_SHAPE(C, batch_size, n_groups, dstate,
                !is_complex ? seqlen : seqlen * 2);
    TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
  }

  if (D_.has_value()) {
    auto D = D_.value();
    TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(D.is_cuda());
    TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
    CHECK_SHAPE(D, dim);
  }

  if (delta_bias_.has_value()) {
    auto delta_bias = delta_bias_.value();
    TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(delta_bias.is_cuda());
    TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
    CHECK_SHAPE(delta_bias, dim);
  }

  at::Tensor z, out_z;
  const bool has_z = z_.has_value();
  if (has_z) {
    z = z_.value();
    TORCH_CHECK(z.scalar_type() == input_type);
    TORCH_CHECK(z.is_cuda());
    TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
    CHECK_SHAPE(z, batch_size, dim, seqlen);
    out_z = torch::empty_like(z);
  }

  const int n_chunks = (seqlen + 2048 - 1) / 2048;
  // const int n_chunks = (seqlen + 1024 - 1) / 1024;
  // at::Tensor out = torch::empty_like(u);
  // Right now u has BHL layout and delta has HBL layout, and we want out to
  // have HBL layout
  at::Tensor out = torch::empty_like(delta);
  at::Tensor x;
  x = torch::empty({batch_size, dim, n_chunks, dstate * 2},
                   u.options().dtype(weight_type));

  SSMParamsBase params;
  set_ssm_params_fwd(
      params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
      is_variable_B, is_variable_C, u, delta, A, B, C, out, z, out_z,
      D_.has_value() ? D_.value().data_ptr() : nullptr,
      delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
      x.data_ptr(), has_z, delta_softplus);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)u.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(
      u.scalar_type(), "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(
            A.scalar_type(), "selective_scan_fwd", [&] {
              selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
            });
      });
  std::vector<at::Tensor> result = {out, x};
  if (has_z) {
    result.push_back(out_z);
  }
  return result;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("fwd", &selective_scan_fwd, "Selective scan forward");
// }