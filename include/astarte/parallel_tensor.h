#pragma once

#include "astarte/caconst.h"
#include "astarte/machine_view.h"
#include "astarte/utils/dot/record_formatter.h"
#include "legion.h"
#include <ostream>
#include <unordered_map>

namespace astarte {

class Op;
class CAModel;
class Initializer;

struct ParallelDim {
    static constexpr int UNKNOWN_DEGREE = -1;
    static constexpr int UNKNOWN_INDEX = -2;

    bool operator==(ParallelDim const &rhs) const {
        if (size != rhs.size) {
            return false;
        }
        if (degree != rhs.degree) {
            return false;
        }
        if (parallel_idx != rhs.parallel_idx) {
            return false;
        }
        return true;
    }

    bool operator!=(ParallelDim const &rhs) const {
        if (size != rhs.size) {
            return true;
        }
        if (degree != rhs.degree) {
            return true;
        }
        if (parallel_idx != rhs.parallel_idx) {
            return true;
        }
        return false;
    }

    int size = 0;                       // Actual size of the tensor
    int degree = UNKNOWN_DEGREE;        // Degree of sharding
    int paralel_idx = UNKNOWN_INDEX;    // runtime information, unique id of each
    bool is_replica_dim = false;        // degree of sharding
};

struct ParallelTensorShape {

    ParallelTensorShape() = default;
    ParallelTensorShape(int num_dims, ParallelDim const dims[MAX_TENSOR_DIM],
                        DataType data_type);
    int num_dims;
    ParallelDim dims[MAX_TENSOR_DIM];
    DataType data_type;

    bool operator==(ParallelTensorShape const &other) const;
    bool operator!=(ParallelTensorShape const &other) const;

    RecordFormatter as_dot() const;

    size_t get_piece_size() const;
    bool is_valid() const;

    int get_num_replica_dims() const;
    int get_num_replicas() const;

    std::unordered_map<int, int> get_mv_dim_to_tensor_dim_mapping() const;
    std::unordered_map<int, int> get_tensor_dim_to_mv_dim_mapping() const;
};

std::ostream &operator<<(std::ostream &, ParallelTensorShape const &);

}; // namespace astarte

namespace std {
template <>
struct hash<astarte::ParallelTensorShape> {
    size_t operator()(astarte::ParallelTensorShape const &) const;
};
} // namespace std

namespace astarte {

class CAConfig;

struct ParallelTensorBase {
    static constexpr ParallelTensorBase *NO_TENSOR = nullptr;
    ParallelTensorBase(void) = default;
    ParallelTensorBase(ParallelTensorBase const &rhs);
    void inline_map(CAConfig &config);
    void inline_unmap(CAConfig &config);
    template <typename T>
    T *get_raw_ptr(CAConfig &config);
    void attach_raw_ptr(CAConfig &config, void *raw_ptr, bool column_major);
    void detach_raw_ptr(CAConfig &config);
    bool get_input_sub_tensor(ParallelConfig const &pc, ParallelTensorBase &tensor, OperatorType type);
    bool get_sub_tensor(MachineView const &mv, ParallelTensorBase &subtensor) const;
    bool get_output_sub_tensor(ParallelConfig const &pc, ParallelTensorBase &tensor, OperatorType type);
    size_t get_owner_independent_hash() const;
    size_t get_volume() const;
    size_t get_total_num_parts() const;
    int get_num_replica_dims() const;
    int get_num_replicas() const;
    Legion::Domain get_domain() const;
    bool check_valid() const;
    bool is_valid_machine_view(MachineView const &view) const;
    void print(std::string const &name) const;
    static bool update_parallel_ids(int numdim, ParallelDim *dims);
    template <typename T>
    bool set_tensor(CAModel const *model, T *data, bool get_parameters);
    ParallelTensorShape get_shape() const;

    template <typename T>
    bool tensor_equal(CAConfig &config, ParallelTensorBase &tensor);
    static bool
        tensor_equal_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
    template <int NDIM>
    static bool tensor_equal_task_with_dim(
        Legion::Task const *task,
        std::vector<Legion::PhysicalRegion> const &regions,
        Legion::Context ctx,
        Legion::Runtime *runtime);

private:
    template <typename T>
    bool get_input_sub_tensor_via_mappings(ParallelConfig const &pc,
                                           ParallelTensorBase &tensor) const;
public:
    size_t parallel_tensor_guid = 0;
    int num_dims = 0;
    ParallelDim dims[MAX_TENSOR_DIM];
    DataType data_type = DT_NONE;
    ParameterSyncType sync_type = ParameterSyncType::NONE;
    Initializer *initializer = nullptr;
    Op const *owner_op = nullptr;
    int owner_idx = 0;
    bool create_gradients = false;

    MachineView machine_view = MachineView::NO_VIEW;
    Legion::IndexSpace parallel_is = Legion::IndexSpace::NO_SPACE;
    Legion::LogicalRegion region = Legion::LogicalRegion::NO_REGION, region_grad = Legion::LogicalRegion::NO_REGION;
    Legion::LogicalPartition part = Legion::LogicalPartition::NO_PART, part_grad = Legion::LogicalPartition::NO_PART;
    Legion::PhysicalRegion physical_region;
};

typedef ParallelTensorBase *ParallelTensor;
typedef ParallelTensorBase *ParallelParameter;

}; // namespace astarte