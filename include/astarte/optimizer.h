#ifndef _ASTARTE_OPTIMIZER_H_
#define _ASTARTE_OPTIMIZER_H_

#include "astarte/parallel_tensor.h"
#include "legion.h"

namespace astarte {

class CAModel;
class OpMeta;

class Optimizer {
public:
    Optimizer(CAModel const *_model);
    virtual void init(void) = 0;
    virtual void next(void) = 0;
    virtual void update(const ParallelTensor p) = 0;
    CAModel const *model;
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(CAModel const *_model,
                 double lr = 0.01f,
                 double momentum = 0.0f,
                 bool nesterov = false,
                 double weight_decay = 0.0f);
    void init(void);
    void next(void);
    void update(const ParallelTensor p);
    void set_weight_decay(double _weight_decay);
    static void ps_update_task(Legion::Task const *task,
                               std::vector<Legion::PhysicalRegion> const &regions,
                               Legion::Context ctx,
                               Legion::Runtime *runtime);
    static void ps_update_task_gpu(SGDOptimizer const *op,
                                   float const *w_grad_ptr,
                                   size_t size,
                                   int num_replicas,
                                   float *w_ptr,
                                   float *v_ptr);
#ifdef CA_USE_NCCL
    static void
        nccl_update_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
    static void nccl_update_task_gpu(SGDOptimizer const *op,
                                     OpMeta const *meta,
                                     float const *w_grad_ptr,
                                     size_t size,
                                     float *w_ptr,
                                     float *v_ptr);
#endif

    double lr, momentum;
    bool nesterov;
    double weight_decay;
    ParameterSyncType comm_type;
    std::map<Legion::LogicalRegion, ParallelTensor> v_values;
};


}; // namespace astarte

#endif