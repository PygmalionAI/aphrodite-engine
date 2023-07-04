import time
from typing import Any, List, Optional

from aphrodite.common.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from aphrodite.processing.scheduler import Scheduler
from aphrodite.engine.args_tools import EngineArgs
from aphrodite.engine.ray_tools import DeviceID, initialize_cluster, ray
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import Sequence, SequenceGroup, SequenceStatus
from aphrodite.transformers_utils.tokenizer import detokenize_incrementally, get_tokenizer
from aphrodite.common.utils import Counter
from aphrodite.task_handler.worker import Worker


logger = init_logger(__name__)

class AphroditeEngine:
    """An engine that receives requests and generates text!

    This is the main class for the Aphrodite Engine. It receives requests from clients
    and generates texts from the model. It includes a tokenizer, a language model
    (possibly distributed across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). This class utilizes iteration-level scheduling and efficient memory
    management to maximize the serving throughput.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the comprehensive list
    of arguments, see `EngineArgs`.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        stage_devices: List[List[DeviceID]],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing Aphrodite Engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})"
        )

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = parallel_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(model_config.tokenizer, model_config.tokenizer_mode)
        self.seq_counter = Counter()

        self.workers: List[Worker] = []
        assert len(stage_devices) == 1, "Only support one stage for now"
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = Worker
            if self.parallel_config.worker_use_ray:
                worker_cls = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    resources={node_resource: 1e-3},
                )(worker_cls).remote
            
            worker = worker_cls(
                model_config,
                parallel_config,
                scheduler_config,
                rank,
                distributed_init_method,
            )
            self.workers.append(worker)
        self._init_cache()

        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)
    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")
        
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. Try increasing the `gpu_memory_utilization` when initializing Aphrodite.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "AphroditeEngine":
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Cluster intialization
        distributed_init_method, devices = initialize_cluster(parallel_config)
        engine = cls(*engine_configs, distributed_init_method, devices, log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:

        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seq.append(seq)

        seq_group = SequenceGroup(request_id, seqs, sampling_params, arrival_time)

        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID.
        Args:
            request_id: The ID of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests"""
        return self.scheduler.has_unfinished_seqs()

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be seapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored_seq_groups = self.scheduler.schedule()
        if (not seq_group_metadata_list) and scheduler_outputs.is_empty() and (not ignored_seq_groups):
            return []

        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        seq_groups = self.scheduler.update(output)

        self._decode_sequence(seq_groups)              # Decode the sequence
        self._stop_sequences(seq_groups)                # Stop the sequences that meet the stopping criteria
        self.scheduler.free_finished_seq_groups()       # Free the finished sequence groups.

        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups + ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def _decode_sequence(self, seq_groups: List[SequenceGroup]) -> None:
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                new_token, new_output_text = detokenize_incrementally(
                    self.tokenizer,
                    seq.output_tokens,
                    seq.get_last_token_id(),
                    skip_special_tokens=True,
                )
                seq.output_tokens.append(new_token)
                seq.output_text = new_output_text

    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Stop the finished sequences."""
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        seq.output_text = seq.output_text[:-len(stop_str)]              # Truncate the output text so that the stop string isn't included in the output.
                        self.scheduler.free_seq(seq, SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue
                
                if (seq.get_len() >=
                    self.scheduler.scheduler_config.max_seq_len):
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                if seq.get_output_len() == sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(seq, SequenceStatus.FINISHED_STOPPED)
                        continue

    def _run_workers(
        self,
        method: str,
        get_all_outputs: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
                
    
    
