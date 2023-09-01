#ifndef _ASTARTE_ATTENTION_PARAMS_H
#define _ASTARTE_ATTENTION_PARAMS_H

#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct MultiHeadAttentionParams {
    LayerID layer_guid;
    int embed_dim, num_heads, kdim, vdim;
    float dropout;
    bool bias, add_bias_kv, add_zero_attn;
    
    bool is_valid(std::tuple<ParallelTensorShape, ParallelTensorShape, ParallelTensorShape> const &) const;
};
bool operator==(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::MultiHeadAttentionParams> {
    size_t operator()(astarte::MultiHeadAttentionParams const &) const;
};
} // namespace std
#endif // _ASTARTE_ATTENTION_PARAMS_H