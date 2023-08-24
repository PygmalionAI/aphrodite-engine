#ifndef _CA_LOSS_FUNCTIONS_H_
#define _CA_LOSS_FUNCTIONS_H_

#include "caconst.h"
#include "legion.h"
#include "parallel_tensor.h"

namespace astarte {
class CAModel;

class Loss {
public:
    Loss(std::string const &loss, bool _repl_labels = false);
    Loss(LossType _loss_type, bool _repl_labels = false);

    static void backward_task(Legion::Task const *task,
                              std::vector<Legion::PhysicalRegion> const &regions,
                              Legion::Context ctx,
                              Legion::Runtime *runtime);
    template <int NDIM>
    static void
        backward_task_with_dim(Legion::Task const *task,
                               std::vector<Legion::PhysicalRegion> const &regions,
                               Legion::Context ctx,
                               Legion::Runtime *runtime);
    void backward(CAModel *model,
                  const ParallelTensor logit,
                  const ParallelTensor label);
    template <int NDIM>
    void backward_with_dim(CAModel *model,
                           const ParallelTensor logit,
                           const ParallelTensor label);
    static void sparse_categorical_crossentropy_loss_backward_kernel_wrapper(
        float *logit_grad_ptr,
        float const *logit_ptr,
        int const *label_ptr,
        size_t logit_volume,
        size_t logit_grad_volume,
        int num_samples,
        int num_classes,
        int k,
        float scale_factor);
    static void categorical_crossentropy_loss_backward_kernel_wrapper(
        float *logit_grad_ptr,
        float const *logit_ptr,
        float const *label_ptr,
        size_t logit_volume,
        size_t logit_grad_volume,
	    float scale_factor);
    static void identity_loss_backward_kernel_wrapper(float *loss_grad_ptr,
                                                      float const *loss_ptr,
                                                      size_t loss_volume,
                                                      size_t loss_grad_volume,
                                                      float scale_factor);
public:
    CAModel *model;
    LossType loss_type;
    bool repl_labels;
    // for aggregate_spec: More precitions than labels
    // scale factor for computing the logit gradients
    // normally 1.0f / global_batch_size
    float scale_factor;
};

}; // namespace astarte

#endif

