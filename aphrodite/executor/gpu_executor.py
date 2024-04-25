from typing import Dict, List, Set

from loguru import logger

from aphrodite.lora.request import LoRARequest
from aphrodite.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from aphrodite.common.sequence import SamplerOutput, SequenceGroupMetadata
from aphrodite.common.utils import (
    get_ip,
    get_open_port,
    get_distributed_init_method,
    make_async,
)


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        If speculative decoding is enabled, we instead create the speculative
        worker.
        """
        if self.speculative_config is None:
            self._init_non_spec_worker()
        else:
            self._init_spec_worker()

    def _init_non_spec_worker(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from aphrodite.task_handler.worker import Worker

        assert (self.parallel_config.world_size == 1
                ), "GPUExecutor only supports single GPU."

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = Worker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _init_spec_worker(self):
        """Initialize a SpecDecodeWorker, using a draft model for proposals.
        """
        assert self.speculative_config is not None

        from aphrodite.spec_decode.multi_step_worker import MultiStepWorker
        from aphrodite.spec_decode.spec_decode_worker import SpecDecodeWorker
        from aphrodite.task_handler.worker import Worker

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        target_worker = Worker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=True,
        )

        draft_worker = MultiStepWorker(
            model_config=self.speculative_config.draft_model_config,
            parallel_config=self.speculative_config.draft_parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=True,
        )

        spec_decode_worker = SpecDecodeWorker.from_workers(
            proposer_worker=draft_worker, scorer_worker=target_worker)

        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = spec_decode_worker

        # Load model handled in spec decode worker.
        self.driver_worker.init_device()

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        logger.info(
            f"Minimum concurrency: {num_gpu_blocks * self.cache_config.block_size / self.scheduler_config.max_model_len:.2f}x"  # noqa: E501
        )

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_lookahead_slots: int,
    ) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=num_lookahead_slots,
        )
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_lookahead_slots: int,
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=num_lookahead_slots,
        )
        return output
