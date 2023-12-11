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

  

}