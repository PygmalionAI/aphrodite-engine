#ifndef _ASTARTE_MODEL_H_
#define _ASTARTE_MODEL_H_
#include "accessor.h"
#include "config.h"
#include "device.h"
#include "inference.h"
#include "memory_optimization.h"
#include "node.h"
#include "operator_params.h"
#include "astarte/utils/hash_utils.h"
#include "astarte/utils/tuple.h"
#include "initializer.h"
#include "layer.h"
#include "legion.h"
#include "loss_functions.h"
#include "metrics_functions.h"
#include "optimizer.h"
#include "parallel_tensor.h"
#include "recompile.h"
#include "runtime.h"
#include "simulator.h"
#include "tensor.h"
#include "tl/optional.hpp"
#include <functional>
#include <unistd.h>
#include <utility>

#include "caconst.h"
#include "catype.h"

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FF_INIT_TASK_ID,
  IMAGE_INIT_TASK_ID,
  LABEL_INIT_TASK_ID,
  LOAD_IMAGES_TASK_ID,
  NORMALIZE_IMAGES_TASK_ID,
  ELEMENTBINARY_INIT_TASK_ID,
  ELEMENTBINARY_INF_TASK_ID,
  ELEMENTBINARY_FWD_TASK_ID,
  ELEMENTBINARY_BWD_TASK_ID,
  ELEMENTUNARY_INIT_TASK_ID,
  ELEMENTUNARY_FWD_TASK_ID,
  ELEMENTUNARY_INF_TASK_ID,
  ELEMENTUNARY_BWD_TASK_ID,
  EXPERTS_INIT_TASK_ID,
  EXPERTS_FWD_TASK_ID,
  EXPERTS_BWD_TASK_ID,
  EXPERTS_INF_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_INIT_PARA_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  CONV2D_UPD_TASK_ID,
  DROPOUT_INIT_TASK_ID,
  DROPOUT_FWD_TASK_ID,
  DROPOUT_BWD_TASK_ID,
  EMBED_INIT_TASK_ID,
  EMBED_FWD_TASK_ID,
  EMBED_BWD_TASK_ID,
  GATHER_INIT_TASK_ID,
  GATHER_FWD_TASK_ID,
  GATHER_BWD_TASK_ID,
  GROUP_BY_INIT_TASK_ID,
  GROUP_BY_FWD_TASK_ID,
  GROUP_BY_BWD_TASK_ID,
  CACHE_INIT_TASK_ID,
  CACHE_FWD_TASK_ID,
  CACHE_UPDATE_TASK_ID,
  CAST_INIT_TASK_ID,
  CAST_FWD_TASK_ID,
  CAST_BWD_TASK_ID,
  AGGREGATE_INIT_TASK_ID,
  AGGREGATE_FWD_TASK_ID,
  AGGREGATE_BWD_TASK_ID,
  AGG_SPEC_INIT_TASK_ID,
  AGG_SPEC_FWD_TASK_ID,
  AGG_SPEC_BWD_TASK_ID,
  POOL2D_INIT_TASK_ID,
  POOL2D_FWD_TASK_ID,
  POOL2D_BWD_TASK_ID,
  BATCHNORM_INIT_TASK_ID,
  BATCHNORM_INIT_PARA_TASK_ID,
  BATCHNORM_FWD_TASK_ID,
  BATCHNORM_BWD_TASK_ID,
  BATCHMATMUL_INIT_TASK_ID,
  BATCHMATMUL_FWD_TASK_ID,
  BATCHMATMUL_BWD_TASK_ID,
  LAYERNORM_INIT_TASK_ID,
  LAYERNORM_FWD_TASK_ID,
  LAYERNORM_INF_TASK_ID,
  LAYERNORM_BWD_TASK_ID,
  RESIDUAL_LAYERNORM_INIT_TASK_ID,
  RESIDUAL_LAYERNORM_INF_TASK_ID,
  ADD_BIAS_RESIDUAL_LAYERNORM_INIT_TASK_ID,
  ADD_BIAS_RESIDUAL_LAYERNORM_INF_TASK_ID,
  SIGMOID_SILU_MULTI_INIT_TASK_ID,
  SIGMOID_SILU_MULTI_INF_TASK_ID,
  LINEAR_INIT_TASK_ID,
  LINEAR_INIT_PARA_TASK_ID,
  LINEAR_INF_TASK_ID,
  LINEAR_FWD_TASK_ID,
  LINEAR_BWD_TASK_ID,
  LINEAR_BWD2_TASK_ID,
  LINEAR_UPD_TASK_ID,
  FLAT_INIT_TASK_ID,
  FLAT_FWD_TASK_ID,
  FLAT_BWD_TASK_ID,
  SOFTMAX_INIT_TASK_ID,
  SOFTMAX_FWD_TASK_ID,
  SOFTMAX_BWD_TASK_ID,
  SOFTMAX_INF_TASK_ID,
  CONCAT_INIT_TASK_ID,
  CONCAT_FWD_TASK_ID,
  CONCAT_BWD_TASK_ID,
  SPLIT_INIT_TASK_ID,
  SPLIT_FWD_TASK_ID,
  SPLIT_BWD_TASK_ID,
  REDUCE_INIT_TASK_ID,
  REDUCE_FWD_TASK_ID,
  REDUCE_BWD_TASK_ID,
  RESHAPE_INIT_TASK_ID,
  RESHAPE_FWD_TASK_ID,
  RESHAPE_BWD_TASK_ID,
  REVERSE_INIT_TASK_ID,
  REVERSE_FWD_TASK_ID,
  REVERSE_BWD_TASK_ID,
  TOPK_INIT_TASK_ID,
  TOPK_FWD_TASK_ID,
  TOPK_BWD_TASK_ID,
  ARG_TOPK_INIT_TASK_ID,
  ARG_TOPK_INF_TASK_ID,
  SAMPLING_INIT_TASK_ID,
  SAMPLING_INF_TASK_ID,
  ARGMAX_INIT_TASK_ID,
  ARGMAX_BEAM_INF_TASK_ID,
  ARGMAX_NORM_INF_TASK_ID,
  TRANSPOSE_INIT_TASK_ID,
  TRANSPOSE_FWD_TASK_ID,
  TRANSPOSE_BWD_TASK_ID,
  ATTENTION_INIT_TASK_ID,
  ATTENTION_FWD_TASK_ID,
  ATTENTION_BWD_TASK_ID,
  RMSNORM_INIT_TASK_ID,
  RMSNORM_FWD_TASK_ID,
  RMSNORM_INF_TASK_ID,
  RESIDUAL_RMSNORM_INIT_TASK_ID,
  RESIDUAL_RMSNORM_INF_TASK_ID,
  BEAM_TOPK_INIT_TASK_ID,
  BEAM_TOPK_INF_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_FWD_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_BWD_TASK_ID,
  INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
  SPEC_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
  SPEC_INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
  TREE_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
  TREE_INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
  MSELOSS_BWD_TASK_ID,
  FUSEDOP_INIT_TASK_ID,
  FUSEDOP_FWD_TASK_ID,
  FUSEDOP_BWD_TASK_ID,
  FUSEDOP_INF_TASK_ID,
  NOOP_INIT_TASK_ID,
  // Metrics tasks
  METRICS_COMP_TASK_ID,
  UPDATE_METRICS_TASK_ID,
  // Parameter server prefetch task
  PS_PREFETCH_TASK_ID,
  // Loss
  LOSS_BWD_TASK_ID,
  // Optimizer with PS
  SGD_UPD_PS_TASK_ID,
  ADAM_UPD_PS_TASK_ID,
  // Optimizer with NCCL
  SGD_UPD_NCCL_TASK_ID,
  ADAM_UPD_NCCL_TASK_ID,
  // Initializer
  GLOROT_INIT_TASK_ID,
  ZERO_INIT_TASK_ID,
  CONSTANT_INIT_TASK_ID,
  UNIFORM_INIT_TASK_ID,
  NORMAL_INIT_TASK_ID,
  // NCCL tasks
  NCCL_GETUNIQUEID_TASK_ID,
  NCCL_INIT_COMMS_TASK_ID,
  // Search
  STRATEGY_SEARCH_TASK_ID,
  // Graph
  GRAPH_OPTIMIZE_TASK_ID,
  // Python data loader
  PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT32_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT64_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT32_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT64_INDEX_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_LOAD_BATCH_GPU_TASK_ID,
  PY_DL_INT32_LOAD_BATCH_GPU_TASK_ID,
  PY_DL_INT64_LOAD_BATCH_GPU_TASK_ID,
  // Parallel Ops
  REPARTITION_INIT_TASK_ID,
  REPARTITION_FWD_TASK_ID,
  REPARTITION_BWD_TASK_ID,
  COMBINE_INIT_TASK_ID,
  COMBINE_FWD_TASK_ID,
  COMBINE_BWD_TASK_ID,
  REPLICATE_INIT_TASK_ID,
  REPLICATE_FWD_TASK_ID,
  REPLICATE_BWD_TASK_ID,
  REDUCTION_INIT_TASK_ID,
  REDUCTION_FWD_TASK_ID,
  REDUCTION_BWD_TASK_ID,
  PIPELINE_INIT_TASK_ID,
  PIPELINE_FWD_TASK_ID,
  PIPELINE_BWD_TASK_ID,
  ALLREDUCE_INIT_TASK_ID,
  ALLREDUCE_INF_TASK_ID,
  ALLREDUCE_FWD_TASK_ID,
  ALLREDUCE_BWD_TASK_ID,
  FUSED_PARALLELOP_INIT_TASK_ID,
  FUSED_PARALLELOP_FWD_TASK_ID,
  FUSED_PARALLELOP_BWD_TASK_ID,
  // InferenceManager & RequestManager
  RM_LOAD_TOKENS_TASK_ID,
  RM_LOAD_POSITION_TASK_ID,
  RM_PREPARE_NEXT_BATCH_TASK_ID,
  RM_PREPARE_NEXT_BATCH_INIT_TASK_ID,
  RM_PREPARE_NEXT_BATCH_BEAM_TASK_ID,
  RM_PREPARE_NEXT_BATCH_VERIFY_TASK_ID,
  // Custom tasks
  CUSTOM_GPU_TASK_ID_FIRST,
  CUSTOM_GPU_TASK_ID_1,
  CUSTOM_GPU_TASK_ID_2,
  CUSTOM_GPU_TASK_ID_3,
  CUSTOM_GPU_TASK_ID_4,
  CUSTOM_GPU_TASK_ID_5,
  CUSTOM_GPU_TASK_ID_6,
  CUSTOM_GPU_TASK_ID_7,
  CUSTOM_GPU_TASK_ID_8,
  CUSTOM_GPU_TASK_ID_LAST,
  CUSTOM_CPU_TASK_ID_FIRST,
  CUSTOM_CPU_TASK_ID_1,
  CUSTOM_CPU_TASK_ID_2,
  CUSTOM_CPU_TASK_ID_3,
  CUSTOM_CPU_TASK_ID_4,
  CUSTOM_CPU_TASK_ID_5,
  CUSTOM_CPU_TASK_ID_6,
  CUSTOM_CPU_TASK_ID_7,
  CUSTOM_CPU_TASK_ID_LAST,
  // Make sure PYTHON_TOP_LEVEL_TASK_ID is
  // consistent with python/main.cc
  PYTHON_TOP_LEVEL_TASK_ID = 11111,
  // Tensor Equal Task
  TENSOR_EQUAL_TASK_ID,
};

enum ShardingID {
    DataParallelShardingID = 135;
};

namespace PCG {
class SearchHelper;
class GraphSearchHelper;
class Graph;
}; // namespace PCG

class CAModel;
class ParallelOp;

void solve_parallel_dim_mappings(
    std::vector<ParallelDimMappingRecord> const &mapping,
    std::vector<ParallelDim const *> const &inputs,
    std::vector<ParallelDim *> const &weights,
    std::vector<ParallelDim *> const &outputs);
std::unordered_map<int, int> output_to_input_mapping(
    std::vector<ParallelDimMappingRecord> const &mapping);
std::unordered_map<int, int> input_to_output_mapping(
    std::vector<ParallelDimMappingRecord> const &mapping);

class NoOp;

ParallelConfig get_basic_data_parallel_config(int num_parts, int dims);

class Aggregate;
class AggregateSpec;
class BatchMatmul;
class Cast;
class Concat;
class Conv2D;
class Dropout;
class ElementBinary;
class ElementUnary;
class Embedding;
class Experts;
class Flat;
class Gather;
class Group_by;
class LayerNorm;
class ResidualLayerNorm;
class AddBiasResidualLayerNorm;
class SigmoidSiluMulti;
class Linear;
class MultiHeadAttention;
class IncMultiHeadSelfAttention;
class TreeIncMultiHeadSelfAttention;
class Pool2D;
class Reduce;
class Reshape;
class Softmax;
class Split;
class TopK;
class ArgTopK;
class Transpose;
class RMSNorm;
class ResidualRMSNorm;
class BeamTopK;
class SpecIncMultiHeadSelfAttention;
class Sampling;
class ArgMax;
class Combine;
class Repartition;
class Reduction;
class Replicate;
class AllReduce;
class FusedParallelOp;
class ParallelOpInfo;

/**
 * This is used to create a type that recursively replaces value type
 * ParallelTensor by ParallelTensorShape in T. e.g. ToShape<std::tuple<int,
 * ParallelTensor>>::type gives std::tuple<int, ParallelTensorShape>
*/
template <typename T>
struct ToShape {
    using type = T;
};

