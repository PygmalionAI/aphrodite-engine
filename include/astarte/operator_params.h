#ifndef _OPERATOR_PARAMS_H
#define _OPERATOR_PARAMS_H

#include "astarte/ops/add_bias_residual_layer_norm_params.h"
#include "astarte/ops/aggregate_params.h"
#include "astarte/ops/aggregate_spec_params.h"
#include "astarte/ops/arg_topk_params.h"
#include "astarte/ops/argmax_params.h"
#include "astarte/ops/attention_params.h"
#include "astarte/ops/batch_matmul_params.h"
#include "astarte/ops/beam_topk_params.h"
#include "astarte/ops/cast_params.h"
#include "astarte/ops/concat_params.h"
#include "astarte/ops/conv_2d_params.h"
#include "astarte/ops/dropout_params.h"
#include "astarte/ops/element_binary_params.h"
#include "astarte/ops/element_unary_params.h"
#include "astarte/ops/element_binary_params.h"
#include "astarte/ops/experts_params.h"
#include "astarte/ops/flat_params.h"
#include "astarte/ops/gather_params.h"
#include "astarte/ops/groupby_params.h"
#include "astarte/ops/inc_multihead_self_attention_params.h"
#include "astarte/ops/layer_norm_params.h"
#include "astarte/ops/linear_params.h"
#include "astarte/ops/pool_2d_params.h"
#include "astarte/ops/reduce_params.h"
#include "astarte/ops/reshape_params.h"
#include "astarte/ops/residual_layer_norm_params.h"
#include "astarte/ops/residual_rms_norm_params.h"
#include "astarte/ops/rms_norm_params.h"
#include "astarte/ops/sampling_params.h"
#include "astarte/ops/sigmoid_silu_multi_params.h"
#include "astarte/ops/softmax_params.h"
#include "astarte/ops/spec_inc_multihead_self_attention_params.h"
#include "astarte/ops/split_params.h"
#include "astarte/ops/topk_params.h"
#include "astarte/ops/transpose_params.h"
#include "astarte/ops/tree_inc_multihead_self_attention_params.h"
#include "astarte/parallel_ops/allreduce_params.h"
#include "astarte/parallel_ops/combine_params.h"
#include "astarte/parallel_ops/fused_parallel_op_params.h"
#include "astarte/parallel_ops/partition_params.h"
#include "astarte/parallel_ops/reduction_params.h"
#include "astarte/parallel_ops/replicate_params.h"
#include "mpark/variant.hpp"

namespace mp = mpark;

namespace astarte {

using OperatorParams = mp::variant<AggregateParams,
                                   AggregateSpecParams,
                                   BatchMatmulParams,
                                   Conv2DParams,
                                   ConcatParams,
                                   CastParams,
                                   ElementBinaryParams,
                                   ElementUnaryParams,
                                   DropoutParams,
                                   EmbeddingParams,
                                   Group_byParams,
                                   LayerNormParams,
                                   ResidualLayerNormParams,
                                   AddBiasResidualLayerNormParams,
                                   SigmoidSiluMultiParams,
                                   LinearParams,
                                   MultiHeadAttentionParams,
                                   IncMultiHeadSelfAttentionParams,
                                   BeamTopKParams,
                                   SpecIncMultiHeadSelfAttentionParams,
                                   TreeIncMultiHeadSelfAttentionParams,
                                   RMSNormParams,
                                   ResidualLayerNormParams,
                                   Pool2DParams,
                                   ReduceParams,
                                   ReshapeParams,
                                   SplitParams,
                                   TopKParams,
                                   ArgTopKParams,
                                   SamplingParams,
                                   ArgMaxParams,
                                   SoftmaxParams,
                                   TransposeParams,
                                   RepartitionParams,
                                   ReplicateParams,
                                   ReductionParams,
                                   CombineParams,
                                   AllReduceParams,
                                   FusedParallelOpParams>;

tl::optional<OperatorParameters> get_op_parameters(Op const *op);
                                   
}; namespace astarte

#endif
