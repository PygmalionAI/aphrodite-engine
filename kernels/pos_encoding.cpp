/*
GPT-NeoX rotary embedding. Credits to EleutherAI.
*/

#include <torch/extension.h>

void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int head_size,
    torch::Tensor& cos_sin_cache);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rotary_embedding",
        &rotary_embedding,
        "Apply rotary embedding to query/key vectors."
    )
}