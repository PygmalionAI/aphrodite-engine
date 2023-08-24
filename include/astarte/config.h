#ifndef _ASTARTE_CONFIG_H_
#define _ASTARTE_CONFIG_H_
#include "caconst.h"
#include "legion.h"
#include <cstring>
#if defined(CA_USE_CUDA) || defined(CA_USE_HIP_CUDA)
#include <cublas_v2.h>
#elif defined(CA_USE_HIP_ROCM)
#include <hipblas.h>
#include <miopen/miopen.h>
#else
#error "Unknown device"
#endif
#include "tl/optional.hpp"
#ifdef CA_USE_NCCL
#include <nccl.h>
#endif

namespace astarte {

#define MAX_NUM_INPUTS 2048
#define MAX_NUM_WEIGHTS 2048
#define MAX_NUM_OUTPUTS 2048
#define MAX_NUM_FUSED_OPERATORS 2048
#define MAX_NUM_FUSED_TENSORS 2048
#define MAX_NUM_WORKERS 1024
#define MAX_FILENAME 200
#define MAX_OPNAME 128
#define MAX_NUM_TRANSFORMER_LAYERS 100
#define MAX_SAMPLES_PER_LOAD 64
#define MAX_FILE_LENGTH 128
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

// distributed
#ifdef CA_USE_NCCL
constexpr ParameterSyncType CHOSEN_SYNC_TYPE = ParameterSyncType::NCCL;
#else
constexpr ParameterSyncType CHOSEN_SYNC_TYPE = ParameterSyncType::PS;
#endif

class CAConfig;

struct CAHandler {
#if defined(CA_USE_CUDA) || defined(CA_USE_HIP_CUDA)
    cudnnHandle_t dnn;
    cublasHandle_t blas;
#else
    miopenHandle_t dnn;
    hipblasHandle_t blas;
#endif
    void *workSpace;
    size_t workSpaceSize;
    void *offload_reserve_space;
    size_t offload_reserve_space_size;
    DataType quantization_type;
    bool allowTensorOpMathConversion;
#ifdef CA_USE_NCCL
    ncclComm_t ncclComm;
#endif
};

struct CAInitInfo {
    size_t workSpaceSize;
    size_t offload_reserve_space_size;
    DataType quantization_type;
    bool allowTensorOpMathConversion;
};

class CAConfig {
public:
    enum PreservedIDs {
        InvalidID = 0,
        DataParallelism_GPU = 1,
        DataParallelism_CPU = 11,
    };

    CAConfig();

    void parse_args(char **argv, int argc);
    static Legion::MappingTagID get_hash_id(std::string const &pcname);

public:
    int epochs, batchSize, printFreq;
    int numNodes, cpusPerNode, workersPerNode;
    float device_mem; 
    float learningRate, weightDecay,
    size_t workSpaceSize;
    Legion::Context lg_ctx;
    Legion::Runtime *lg_hlr;
    bool syntheticInput, profiling, perform_fusion;
    size_t simulator_work_space_size;
    size_t search_budget;
    float search_alpha;
    bool search_overlap_backward_update;
    CompMode computationMode;
    bool cpu_offload;
    size_t offload_reserve_space_size;
    DataType quantization_type;
    // Control parallelizable dimensions
    bool only_data_parallel;
    bool enable_sample_parallel;
    bool enable_parameter_parallel;
    bool enable_attribute_parallel;
    bool enable_inplace_optimizations;
    // Control parallelsim degrees in inference
    int data_parallelism_degree;
    int tensor_parallelism_degree;
    int pipeline_parallelism_degree;
    // Control Tensor Op Math Conversion
    bool allow_tensor_op_math_conversion;
    std::string dataset_path;
    std::string import_strategy_file;
    std::string export_strategy_file;
    std::string export_strategy_task_graph_file;
    std::string export_strategy_computation_graph_file;
    bool include_costs_dot_graph;
    tl::optional<std::string> substitution_json_path = tl::nullopt; // uh?
    int machine_model_version;
    std::string machine_model_file;
    int simulator_segment_size;
    int simulator_max_num_segments;
    bool enable_propagation;
    tl::optional<int> search_num_nodes = tl::nullopt;
    tl::optional<int> search_num_workers = tl::nullopt;
    int base_optimize_threshold;
    bool enable_control_replication;
    int python_data_loader_type;
    bool perform_memory_search{false};
};

class CAIterationConfig {
public:
    CAIterationConfig();
    void reset();
    int seq_length;
};

enum FieldIDs {
    FID_DATA,
};

}; // namespace astarte

#endif //_ASTARTE_CONFIG_H_