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

template <>
struct ToShape<ParallelTensor> {
    using type = ParallelTensorShape;
};

template <typename .. Args, template <typename...> class Container>
struct ToShape<Container<Args...>> {
    using type = Container<typename ToShape<Args>::type...>;
};

// TODO: Move this to an appropriate place
template <typename Input>
typename ToShape<Input>::type get_input_shape(Input const &input) = delete;

template <>
std::tuple<> get_input_shape(std::tuple<> const &);

template <>
std::tuple<ParallelTensorShape, ParallelTensorShape, ParallelTensorShape>
  get_input_shape(std::pair<ParallelTensor, ParallelTensor> const &inputs);

template <>
ParallelTensorShape get_input_shape(ParallelTensor const &input);

template <>
std::pair<ParallelTensorShape, ParallelTensorShape>
  get_input_shape(std::pair<ParallelTensor, ParallelTensor> const &inputs);

template <>
std::vector<ParallelTensorShape>
  get_input_shape(std::vector<ParallelTensor> const &inputs);

class CAModel {
public:
  CAModel(CAConfig &config, bool cpu_offload = false);

  static constexpr float PROPAGATION_CHANCE = 0.25;
  static constexpr float CONTINUE_PROPAGATION_CHANCE = 0.75;
  static constexpr float PROPAGATION_SIZE_WEIGHT = 1.0;

  bool cpu_offload;
  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(const Tensor x, char const *name = NULL);
  // Add an add layer
  Tensor add(const Tensor x,
             const Tensor y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a subtract layer
  Tensor subtract(const Tensor x,
                  const Tensor y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a multiply layer
  Tensor multiply(const Tensor x,
                  const Tensor y,
                  bool inplace_a = false,
                  char const *name = NULL);
  // Add a divide layer
  Tensor divide(const Tensor x,
                const Tensor y,
                bool inplace_a = false,
                char const *name = NULL);
  // Add a max layer
  Tensor max(const Tensor x,
             const Tensor y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a min layer
  Tensor min(const Tensor x,
             const Tensor y,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a rsqrt layer
  Tensor rsqrt(const Tensor x, bool inplace = True, char const *name = NULL);
  // Add a pow layer
  Tensor pow(const Tensor x,
             float const exponent,
             bool inplace_a = false,
             char const *name = NULL);
  // Add a scalar multiply layer
  Tensor scalar_multiply(const Tensor x,
                         float const scalar,
                         bool inplace = false,
                         char const *name = NULL);
  // Add a scalar add layer
  Tensor scalar_add(const Tensor x,
                    float const scalar,
                    bool inplace = false,
                    char const *name = NULL);
  // Add a scalar subtract layer
  Tensor scalar_sub(const Tensor x,
                    float const scalar,
                    bool inplace = false,
                    char const *name = NULL);
  // Add a scalar truediv layer
  Tensor scalar_truediv(const Tensor x,
                        float const scalar,
                        bool inplace = false,
                        char const *name = NULL);
  // Add a sin layer
  Tensor sin(const Tensor x, char const *name = NULL);
  // Add a cos layer
  Tensor cos(const Tensor x, char const *name = NULL);
  // Add activation layers
  Tensor relu(const Tensor x, bool inplace = true, char const *name = NULL);
  Tensor identity(const Tensor x, char const *name = NULL);
  Tensor gelu(const Tensor x, char const *name = NULL);
  Tensor sigmoid(const Tensor x, char const *name = NULL);
  Tensor tanh(const Tensor x, char const *name = NULL);
  Tensor elu(const Tensor x, char const *name = NULL);
  // Add a 2D convolution layer
  Tensor conv2d(const Tensor input,
                int outChannels,
                int kernelH,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                ActiMode activation = AC_MODE_NONE,
                int groups = 1,
                bool use_bias = true,
                Layer const *shared_op = NULL,
                Initializer *kernel_initializer = NULL,
                Initializer *bias_initializer = NULL,
                char const *name = NULL);
  // Add a dropout layer
  Tensor dropout(const Tensor input,
                 float rate,
                 unsigned long long seed = 0,
                 char const *name = NULL);
  // Add an embedding layer
  Tensor embedding(const Tensor input,
                   int num_entries,
                   int outDim,
                   AggrMode aggr,
                   DataType dtype = DT_FLOAT,
                   Layer const *shared_op = NULL,
                   Initializer *kernel_initializer = NULL,
                   char const *name = NULL);
  // Add a gather layer
  Tensor gather(const Tensor input,
                const Tensor index,
                int dim,
                char const *name = NULL);
  // Add a group_by layer
  Tensor group_by(const Tensor data,
                  const Tensor assign,
                  Tensor *outputs,
                  int n,
                  float alpha,
                  char const *name = NULL);
  // Add a cache layer
  Tensor cache(Tensor const &input,
               int num_batches,
               std::function<float(float *, void const *, void const *, int)>
                   score_f = {},
               char const *name = NULL);
  // Add aggregate layer
  Tensor aggregate(Tensor const *inputs,
                   int n,
                   float lambda_bal,
                   char const *name = NULL);
  // Add aggregate_spec layer
  Tensor aggregate_spec(Tensor const *inputs,
                        int n,
                        float lambda_bal,
                        char const *name = NULL);
  // Add a pool2d layer
  Tensor pool2d(const Tensor input,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE,
                char const *name = NULL);
  // Add a layer_norm layer
  Tensor layer_norm(const Tensor input,
                    std::vector<int> const &axes,
                    bool elementwise_affine,
                    float eps,
                    bool use_bias = true,
                    DataType data_type = DT_NONE,
                    char const *name = NULL);
  // Add a layer_norm layer with residual(s)
  void residual_layer_norm(const Tensor input,
                           const Tensor residual1,
                           const Tensor residual2,
                           Tensor *outputs,
                           bool use_two_residuals,
                           std::vector<int> const &axes,
                           bool elementwise_affine,
                           float eps,
                           bool use_bias = true,
                           DataType data_type = DT_NONE,
                           char const *name = NULL);
  // Add a add_bias_residual_layer_norm layer
  void add_bias_residual_layer_norm(const Tensor input,
                                    const Tensor residual,
                                    Tensor *outputs,
                                    std::vector<int> const &axes,
                                    bool elementwise_affine,
                                    float eps,
                                    bool use_bias = true,
                                    DataType data_type = DT_NONE,
                                    char const *name = NULL);
  // Add a sigmoid_silu_multi layer
  Tensor sigmoid_silu_multi(const Tensor input1,
                            const Tensor input2,
                            DataType data_type = DT_NONE,
                            char const *name = NULL);
  // Add a batch_norm layer
  Tensor batch_norm(const Tensor input, bool relu = true, char const *name = NULL);
  // Add a batch_matmul layer
  Tensor batch_matmul(const Tensor A,
                      const Tensor B,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      char const *name = NULL);
  // Add a root mean square layer
  Tensor rms_norm(const Tensor input,
                  float eps,
                  int dim,
                  DataType data_type = DT_NONE,
                  char const *name = NULL);
  // Add a residual root mean square layer
  void residual_rms_norm(const Tensor input1,
                         const Tensor input2,
                         Tensor *outputs,
                         float eps,
                         int dim,
                         DataType data_type = DT_NONE,
                         char const *name = NULL);
  // Add a beam search top_k layer
  Tensor beam_top_k(const Tensor input,
                    int max_beam_size,
                    bool sorted,
                    char const *name = NULL);
  // Add a dense layer
  Tensor dense(const Tensor input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               DataType data_type = DT_NONE,
               Layer const *shared_op = NULL,
               Initializer *kernel_initializer = NULL,
               Initializer *bias_initializer = NULL,
               RegularizerMode regularizer_mode = REG_MODE_NONE,
               float regularizer_lambda = 0.0,
               char const *name = NULL);
  // Add a cast layer
  Tensor cast(const Tensor input, DataType dtype, char const *name = NULL);
  // Add a concat layer
  Tensor concat(int n, Tensor const *tensors, int axis, char const *name = NULL);
  // Add an experts layer
  Tensor experts(
    Tensor const *inputs,
    int num_experts,
    int experts_start_idx,
    int experts_output_dim_size,
    float alpha,
    int experts_num_layers = 1,           // number of linear layers per expert
    int experts_internal_dim_size = 0,    // hidden dimension for internal layers
    char const *name = NULL);
  
  // Add a mean layer
  Tensor mean(const Tensor input,
              std::vector<int> const &dims,
              bool keepdims,
              char const *name);
  // Add a MoE layer (wrapping top_k, group_by, and aggregate operators)
  Tensor moe(const Tensor input,
             int num_exp,
             int num_select,
             int expert_hidden_size,
             float alpha,
             float lambda);
  // Add a split layer
  void split(const Tensor input,
             Tensor *outputs,
             std::vector<int> const &split,
             int axis,
             char const *name = NULL);
  // Add a flat layer
  Tensor flat(const Tensor input, char const *name = NULL);
  // Add a softmax layer
  Tensor softmax(const Tensor input,
                 int dim = -1,
                 DataType data_type = DT_NONE,
                 char const *name = NULL);
  // Create input tensor and constants
  Tensor transpose(const Tensor input,
                   std::vector<int> const &perm,
                   char const *name = NULL);
  Tensor reduce_sum(const Tensor input,
                    std::vector<int> const &axes,
                    bool keepdims = false,
                    char const *name = nullptr);
  Tensor reshape(const Tensor input,
                 std::vector<int> const &shape,
                 char const *name = NULL);
  Tensor reverse(const Tensor input, int axis, char const *name = NULL);
  void top_k(const Tensor input,
             Tensor *outputs,
             int k,
             bool sorted,
             char const *name = NULL);
  Tensor arg_top_k(const Tensor input,
                   int k,
                   bool sorted,
                   char const *name = NULL);
  Tensor argmax(const Tensor input, bool beam_search, char const *name = NULL);
  Tensor sampling(const Tensor input, float top_p, char const *name = NULL);
  Tensor multihead_attention(const Tensor query,
                             const Tensor key,
                             const Tensor value,
                             int embed_dim,
                             int num_heads,
                             int kdim = 0,
                             int vdim = 0,
                             float dropout = 0.0f,
                             bool bias = true,
                             bool add_bias_kv = false,
                             bool add_zero_attn = false,
                             DataType data_type = DT_NONE,
                             Initializer *kernel_initializer = NULL,
                             char const *name = NULL);
  Tensor inc_multihead_self_attention(const Tensor input,
                                      int embed_dim,
                                      int num_heads,
                                      int kdim = 0,
                                      int vdim = 0,
                                      float dropout = 0.0f,
                                      bool bias = false,
                                      bool add_bias_kv = false,
                                      bool add_zero_attn = false,
                                      DataType data_type = DT_NONE,
                                      Initializer *kernel_initializer = NULL,
                                      bool apply_rotary_embedding = false,
                                      bool scaling_query = false,
                                      float scaling_factor = 1.0f,
                                      float qk_prod_scaling = true,
                                      bool position_bias = false,
                                      char const *name = NULL);
  Tensor spec_inc_multihead_self_attention(const Tensor input,
                                           int embed_dim,
                                           int num_heads,
                                           int kdim = 0,
                                           int vdim = 0,
                                           float dropout = 0.0f,
                                           bool bias = false,
                                           bool add_bias_kv = false,
                                           bool add_zero_attn = false,
                                           DataType data_type = DT_NONE,
                                           Initializer *kernel_initializer = NULL,
                                           bool apply_rotary_embedding = false,
                                           bool scaling_query = false,
                                           float scaling_factor = 1.0f,
                                           bool qk_prod_scaling = true,
                                           bool position_bias = false,
                                           char const *name = NULL);
  Tensor inc_multihead_self_attention_verify(
      const Tensor input,
      int embed_dim,
      int num_heads,
      int kdim = 0,
      int vdim = 0,
      float dropout = 0.0f,
      bool bias = false,
      bool add_bias_kv = false,
      bool add_zero_attn = false,
      DataType data_type = DT_NONE,
      Initializer *kernel_initializer = NULL,
      bool apply_rotary_embedding = false,
      bool scaling_query = false,
      float scaling_factor = 1.0f,
      bool qk_prod_scaling = true,
      bool position_bias = false,
      char const *name = NULL);
    
  Tensor inc_multiquery_self_attention(const Tensor input,
                                       int embed_dim,
                                       int num_q_heads,
                                       int num_kv_heads,
                                       int kdim = 0,
                                       int vdim = 0,
                                       float dropout = 0.0f,
                                       bool bias = false,
                                       bool add_bias_kv = false,
                                       bool add_zero_attn = false,
                                       DataType data_type = DT_NONE,
                                       Initializer *kernel_initializer = NULL,
                                       bool apply_rotary_embedding = false,
                                       bool scaling_query = false,
                                       float scaling_factor = 1.0f,
                                       bool qk_prod_scaling = true,
                                       bool position_bias = false,
                                       char const *name = NULL);
  Tensor spec_inc_multiquery_self_attention(const Tensor input,
                                            int embed_dim,
                                            int num_q_heads,
                                            int num_kv_heads,
                                            int kdim = 0,
                                            int vdim = 0,
                                            float dropout = 0.0f,
                                            bool bias = false,
                                            bool add_bias_kv = false,
                                            bool add_zero_attn = false,
                                            DataType data_type = DT_NONE,
                                            Initializer *kernel_initializer = NULL,
                                            bool apply_rotary_embedding = false,
                                            bool scaling_query = false,
                                            float scaling_factor = 1.0f,
                                            bool qk_prod_scaling = true,
                                            bool position_bias = false,
                                            char const *name = NULL);
  Tensor inc_multiquery_self_attention_verify(
      const Tensor input,
      int embed_dim,
      int num_q_heads,
      int num_kv_heads,
      int kdim = 0,
      int vdim = 0,
      float dropout = 0.0f,
      bool bias = false,
      bool add_bias_kv = false,
      bool add_zero_attn = false,
      DataType data_type = DT_NONE,
      Initializer *kernel_initializer = NULL,
      bool apply_rotary_embedding = false,
      bool scaling_query = false,
      float scaling_factor = 1.0f,
      bool qk_prod_scaling = true,
      bool position_bias = false,
      char const *name = NULL);
                
}