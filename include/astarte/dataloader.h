#ifndef __ASTARTE_DATALOADER_H__
#define __ASTARTE_DATALOADER_H__

#include "astarte/model.h"

struct NetConfig {
  NetConfig(void);
  std::string dataset_path;
};

struct DLRMConfig {
  DLRMConfig(void);
  int sparse_feature_size, sigmoid_bot, sigmoid_top, embedding_bag_size;
  float loss_threshold;
  std::vector<int> embedding_size, mlp_bot, mlp_top;
  std::string arch_interaction_op, dataset_path,
};

class SingleDataLoader {
public:
  SingleDataLoader(astarte::CAModel &ca,
                   astarte::ParallelTensor input,
                   astarte::ParallelTensor full_input_,
                   int num_samples_,
                   DataType datatype_);
  SingleDataLoader(astarte::CAModel &ca,
                   astarte::ParallelTensor input,
                   void *full_input_ptr,
                   int num_samples_,
                   DataType datatype_);

  void next_batch(astarte::CAModel &);

  void reset(void);

  static void register_cpu_tasks(Legion::Runtime *runtime = NULL,
                                 bool pre_register = true,
                                 bool enable_control_replication = true);
  static void register_gpu_tasks(Legion::Runtime *runtime = NULL,
                                 bool pre_register = true,
                                 bool enable_control_replication = true);
  
  template <typename DT>
  static void load_input(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime);
  
  template <typename DT>
  static void load_entire_dataset_from_numpy(
    Legion::Task const *task,
    std::vector<Legion::PhysicalRegion> const &regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void load_entire_dataset_from_numpy_with_dim(
    Legion::Task const *task,
    std::vector<Legion::PhysicalRegion> const &regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);
  template <typename DT>
  static void index_load_entire_dataset_from_numpy(
    Legion::Task const *task,
    std::vector<Legion::PhysicalRegion> const &regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);
  template <typename DT, int NDIM>
  static void index_load_entire_dataset_from_numpy_with_dim(
    Legion::Task const *task,
    std::vector<Legion::PhysicalRegion> const &regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

private:
  template <int NDIM>
  void next_batch_xd_launcher(astarte::CAModel &ca, int task_id);

  template <int NDIM>
  void index_loader_xd_launcher(astarte::CAModel &ca,
                                int task_id,
                                void *full_input_ptr,
                                size_t size_per_sample);

public:
  int num_samples, next_index;
  DataType datatype;
  astarte::ParallelTensor full_input, batch_input;

};

#define MAX_NUM_SAMPLES 4196
struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

struct IndexLoadArg {
  int num_samples;
  size_t size_per_sample;
  int idx;
  void *ptr;
};

#endif // __ASTARTE_DATALOADER_H__
