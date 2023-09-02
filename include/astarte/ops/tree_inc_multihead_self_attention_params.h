#ifndef _ASTARTE_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H
#define _ASTARTE_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H

#include "astarte/catype.h"
#include "astarte/parallel_tensor.h"

namespace astarte {

struct TreeIncMultiHeadSelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_q_heads, kdim, vdim, num_kv_heads,
      tensor_parallelism_degree;
  float dropout, scaling_factor;
  bool bias, add_bias_kv, add_zero_attn, apply_rotary_embedding, scaling_query,
      qk_prod_scaling;
  DataType quantization_type;
  bool offload;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(TreeIncMultiHeadSelfAttentionParams const &,
                TreeIncMultiHeadSelfAttentionParams const &);

} // namespace astarte

namespace std {
template <>
struct hash<astarte::TreeIncMultiHeadSelfAttentionParams> {
  size_t
      operator()(astarte::TreeIncMultiHeadSelfAttentionParams const &) const;
};
} // namespace std

#endif // _ASTARTE_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H
