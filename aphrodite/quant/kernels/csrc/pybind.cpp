#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "layernorm/layernorm.h"
#include "quantization/gemm_cuda.h"
#include "pos_embed/pos_encoding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel for quantized LLMs.");
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("rotary_embedding_neox", &rotary_embedding_neox, "Apply GPT-NeoX style rotary embeddings to QK.");
}