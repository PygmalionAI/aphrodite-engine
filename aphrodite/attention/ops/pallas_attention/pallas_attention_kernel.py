# ruff: noqa
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jaxlib
import torch
import torch_xla


def jax_import_guard():
    # Somehow, we need to grab the TPU before JAX locks it.
    # Otherwise, any pt-xla TPU operations will hang.
    torch_xla._XLAC._init_computation_client()


def convert_torch_dtype_to_jax(dtype: torch.dtype) -> "jnp.dtype":
    # Import JAX within the function such that we don't need to call the
    # jax_import_guard() in the global scope which could cause problems
    # for xmp.spawn.
    jax_import_guard()
    import jax.numpy as jnp

    if dtype == torch.float32:
        return jnp.float32
    elif dtype == torch.float64:
        return jnp.float64
    elif dtype == torch.float16:
        return jnp.float16
    elif dtype == torch.bfloat16:
        return jnp.bfloat16
    elif dtype == torch.int32:
        return jnp.int32
    elif dtype == torch.int64:
        return jnp.int64
    elif dtype == torch.int16:
        return jnp.int16
    elif dtype == torch.int8:
        return jnp.int8
    elif dtype == torch.uint8:
        return jnp.uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def to_jax_shape_dtype_struct(tensor: torch.Tensor) -> "jax.ShapeDtypeStruct":
    # Import JAX within the function such that we don't need to call the
    # jax_import_guard() in the global scope which could cause problems
    # for xmp.spawn.
    jax_import_guard()
    import jax

    return jax.ShapeDtypeStruct(tensor.shape,
                                convert_torch_dtype_to_jax(tensor.dtype))


def _extract_backend_config(
        module: "jaxlib.mlir._mlir_libs._mlir.ir.Module") -> Optional[str]:
    """
  This algorithm intends to extract the backend config from the compiler IR like the following,
  and it is not designed to traverse any generic MLIR module.

  module @jit_add_vectors attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
    func.func public @main(%arg0: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<8xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
      %0 = call @add_vectors(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @add_vectors(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @wrapped(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @wrapped(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @apply_kernel(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @apply_kernel(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSMTkuMC4wZ2l0AAErCwEDBQcJAQMLAwUDDQcFDxEJBRMVA3lZDQFVBwsPEw8PCw8PMwsLCwtlCwsLCwsPCw8PFw8LFw8PCxcPCxcTCw8LDxcLBQNhBwNZAQ0bBxMPGw8CagMfBRcdKy0DAycpHVMREQsBBRkVMzkVTw8DCxUXGRsfCyELIyUFGwEBBR0NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFHwUhBSMFJQUnEQMBBSkVLw8dDTEXA8IfAR01NwUrFwPWHwEVO0EdPT8FLRcD9h8BHUNFBS8XA3InAQMDSVcFMR1NEQUzHQ1RFwPGHwEFNSN0cHUubWVtb3J5X3NwYWNlPHZtZW0+ACNhcml0aC5vdmVyZmxvdzxub25lPgAXVQMhBx0DJwMhBwECAgUHAQEBAQECBASpBQEQAQcDAQUDEQETBwMVJwcBAQEBAQEHAwUHAwMLBgUDBQUBBwcDBQcDAwsGBQMFBQMLCQdLRwMFBQkNBwMJBwMDCwYJAwUFBRENBAkHDwURBQABBgMBBQEAxgg32wsdE2EZ2Q0LEyMhHSknaw0LCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAc3RvcmUAL3dvcmtzcGFjZXMvd29yay9weXRvcmNoL3hsYS90ZXN0L3Rlc3Rfb3BlcmF0aW9ucy5weQBhZGRfdmVjdG9yc19rZXJuZWwAZGltZW5zaW9uX3NlbWFudGljcwBmdW5jdGlvbl90eXBlAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAHN5bV9uYW1lAG1haW4AdmFsdWUAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgOCldLCBbXSksKSksICg4LCksICgpKV0sIFtdKSwpKV0AYWRkX3ZlY3RvcnMAdGVzdF90cHVfY3VzdG9tX2NhbGxfcGFsbGFzX2V4dHJhY3RfYWRkX3BheWxvYWQAPG1vZHVsZT4Ab3ZlcmZsb3dGbGFncwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQA=\22, \22needs_layout_passes\22: true}}", kernel_name = "add_vectors_kernel", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
  }

  Basically, what we are looking for is a two level of operations, and the tpu_custom_call operation in the inner level. It will return None if the payload is not found.
  """
    for operation in module.body.operations:
        assert len(operation.body.blocks
                   ) == 1, "The passing module is not compatible."
        for op in operation.body.blocks[0].operations:
            if op.name == "stablehlo.custom_call":
                return op.backend_config.value
    return None


def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 **kwargs):
    # Import JAX within the function such that we don't need to call the
    # jax_import_guard() in the global scope which could cause problems
    # for xmp.spawn.
    jax_import_guard()
    import jax
    import jax._src.pallas.mosaic.pallas_call_registration

    jax_args = []  # for tracing
    tensor_args = []  # for execution
    for i, arg in enumerate(args):
        # TODO: Could the args be a tuple of tensors or a list of tensors? Flattern them?
        if torch.is_tensor(arg):
            # ShapeDtypeStruct doesn't have any storage and thus is very suitable for generating the payload.
            jax_meta_tensor = to_jax_shape_dtype_struct(arg)
            jax_args.append(jax_meta_tensor)
            tensor_args.append(arg)
        else:
            jax_args.append(arg)

    # Here we ignore the kwargs for execution as most of the time, the kwargs is only used in traced code.
    ir = jax.jit(kernel,
                 static_argnums=static_argnums,
                 static_argnames=static_argnames).lower(
                     *jax_args, **kwargs).compiler_ir()
    payload = _extract_backend_config(ir)
    return payload, tensor_args


def paged_attention(q,
                    k_pages,
                    v_pages,
                    lengths,
                    page_indices,
                    pages_per_compute_block,
                    megacore_mode: str = None,
                    attn_logits_soft_cap: float = None):
    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    jax_import_guard()
    from aphrodite.attention.ops.pallas_attention.pallas_attention_kernel_utils import \
        paged_attention  # noqa

    assert megacore_mode in [
        "kv_head", "batch", None
    ], "megacore_mode must be one of ['kv_head', 'batch', None]."

    payload, tensor_args = trace_pallas(
        paged_attention,
        q,
        k_pages,
        v_pages,
        lengths,
        page_indices,
        pages_per_compute_block=pages_per_compute_block,
        megacore_mode=megacore_mode,
        attn_logits_soft_cap=attn_logits_soft_cap,
        static_argnames=[
            "pages_per_compute_block", "megacore_mode", "attn_logits_soft_cap"
        ],
    )

    batch_size, num_heads, head_dim = q.shape
    num_kv_heads, _, page_size, head_dim_k = k_pages.shape
    batch_size_paged_indices, pages_per_sequence = page_indices.shape
    q_dtype_for_kernel_launch = q.dtype
    if (num_heads // num_kv_heads) % 8 != 0:
        q = q.reshape(batch_size, num_heads, 1, head_dim)
        q_dtype_for_kernel_launch = torch.float32

    page_indices_reshaped = page_indices.reshape(-1)
    buffer_index = torch.zeros((1, ), dtype=torch.int32).to("xla")
    step = torch.zeros((1, ), dtype=torch.int32).to("xla")
    output_shape = torch.Size(list(q.shape[:-1]) + [1])

    output, _, _ = torch_xla._XLAC._xla_tpu_custom_call(
        [
            lengths,
            page_indices_reshaped,
            buffer_index,
            step,
            q.to(q_dtype_for_kernel_launch),
            k_pages,
            v_pages,
        ], payload, [q.shape, output_shape, output_shape],
        [q_dtype_for_kernel_launch, torch.float32, torch.float32])

    return output.reshape(batch_size, num_heads, head_dim).to(q.dtype)
