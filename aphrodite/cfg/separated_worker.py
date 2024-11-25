from typing import List, Optional, Tuple

import torch

from aphrodite.common.sequence import IntermediateTensors, SamplerOutput
from aphrodite.distributed import get_pp_group, get_tp_group
from aphrodite.task_handler.model_runner import (
    ModelInputForGPUWithSamplingMetadata)
from aphrodite.task_handler.model_runner_base import BroadcastableModelInput
from aphrodite.task_handler.worker import Worker
from aphrodite.task_handler.worker_base import WorkerInput


class SeparatedWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def get_logits(
        self,
        hidden_or_intermediate_states: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> torch.Tensor:
        return self.model_runner.get_logits(
            hidden_or_intermediate_states, model_input)

    @torch.inference_mode()
    def compute_logits(
        self,
        logits: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> torch.Tensor:
        return self.model_runner.compute_logits(logits, model_input)

    @torch.inference_mode()
    def do_sample(
        self,
        logits: torch.Tensor,
        model_input: ModelInputForGPUWithSamplingMetadata,
    ) -> List[SamplerOutput]:
        return self.model_runner.do_sample(logits, model_input)

    @torch.inference_mode()
    def execute_model_part(
        self,
        inputs: Tuple[BroadcastableModelInput, WorkerInput],
    ) -> Optional[List[SamplerOutput]]:

        model_input, worker_input = inputs
        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group()))

        hidden_or_intermediate_states = self.model_runner.model_execute(
            model_input, 
            self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None, 
            intermediate_tensors,
            num_steps
        )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.get_logits(hidden_or_intermediate_states, model_input)

        return logits
