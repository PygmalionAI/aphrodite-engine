"""A Neuron worker class."""
from typing import List, Optional

import torch
import torch.distributed

from aphrodite.common.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from aphrodite.modeling import set_random_seed
from aphrodite.common.sequence import SamplerOutput, SequenceGroupMetadata
from aphrodite.task_handler.neuron_model_runner import NeuronModelRunner


class NeuronWorker:
    """A worker class that executes the model on a group of neuron cores."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        self.model_runner = NeuronModelRunner(model_config, parallel_config,
                                              scheduler_config, device_config)

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Optional[SamplerOutput]:
        num_seq_groups = len(seq_group_metadata_list)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list)
        return output
