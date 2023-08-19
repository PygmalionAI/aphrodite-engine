#pragma once

#include "astarte/machine_view.h"
#include "legion.h"
#include <unordered_map>

namespace astarte {

class Layer;
class CAModel;
class Initializer;
class ParallelTensorBase;

struct TensorBase {
    TensorBase(void) = default;
    TensorBase(TensorBase const &rhs);
    size_t get_volume() const;
    Legion::Domain get_domain() const;
    void print(std::string const &name) const;

    template <typename T>
    bool set_tensor(CAModel const *model,
                    std::vector<int> const &dims,
                    T const *data);
    template <typename T>
    bool get_tensor(CAConfig const *model, T *data, bool get_gradients);
    template <typename T>
    bool get_output_parallel_tensor(CAConfig const *ff,
                                    T *data,
                                    bool get_gradients);

private:
    // template <typename T>
    // bool get_input_sub_tensor_via_mappings(const ParallelConfig& pc,
    // TensorBase& tensor) const;
public:
    size_t tensor_guid = 0;
    int num_dims = 0;
    int dims[MAX_TENSOR_DIM];
    DataType data_type = DT_NONE;
    ParameterSyncType sync_type = ParameterSyncType::NONE;
    Initializer *initializer = nullptr;
    ParallelTensorBase *parallel_tensor = nullptr;
    Layer const *owner_layer = nullptr;
    int owner_idx = 0;
    bool create_gradients = false;
};

typedef TensorBase *Tensor;
typedef TensorBase *Parameter;

}; // namespace astarte

