#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <layernorm/layernorm.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel for quantized LLMs.");
}