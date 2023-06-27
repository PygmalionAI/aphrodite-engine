import asyncio
import time
from typing import Dict, List, Optional

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.aphrodite import AphroditeEngine
from aphrodite.engine.ray_tools import initialize_cluster, ray
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # in seconds


class AsyncAphrodite:
    """An asynchronous wrapper for Aphrodite.

    This class is used to wrap the AphroditeEngine class to make is asynchronous. It
    uses asyncio to create a background loop that keeps procesing incoming requests.
    The AphroditeEngine is kicked by the generate method when there are requests in the
    waiting queue. The generate method yields the outputs from the AphroditeEngine to
    the caller.

    NOTE: For the comprehensive list of arguments, see `AphroditeEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for distributed
            execution. Should be the same as `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make AphroditeEngine a Ray actor. If so, the async frontend
            will be exeduted in a separate process as the model workers.
        log_requests: Whether to log the requests.
        *args, **kwargs: Arguments for AphroditeEngine. 
    """
    
    def __init__(self, worker_use_ray: bool, engine_use_ray: bool, log_requests: bool = True, *args, **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        if not self.engine_use_ray:
            engine_class = AphroditeEngine
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(AphroditeEngine).remote
        else:
            engine_class = ray.remote(num_gpus=1)(AphroditeEngine).remote
        self.engine = engine_class(*args, **kwargs)
        self.request_outputs: Dict[str, RequestOutput] = {}
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_engine_running = False
        self.kicking_request_id: Optional[str] = None

    
    async def engine_step(self, kicking_request_id: Optional[str] = None):
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            await asyncio.sleep(0)
            request_outputs = self.engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None

        for request_outputs in request_outputs:
            request_id = request_output.request_id
            self.request_outputs[request_id] = request_output
            self.request_events[request_id].set()

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None
    ) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the request
        into the waiting queue of the AphroditeEngine and streams the outputs from the 
        AphroditeEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if `prompt_token_ids` is provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we use the tokenizer
                to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the AphroditeEngine for the request.
        """
        arrival_time = time.time()

        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        if self.log_requests:
            logger.info(f"Received request {request_id}: "
                        f"prompt: {prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {prompt_token_ids}.")

        if self.engine_use_ray:
            await self.engine.add_request.remote(
                request_id, prompt, sampling_params, prompt_token_ids=prompt_token_ids, arrival_time=arrival_time)
        else:
            self.engine.add_request(
                request_id, prompt, sampling_params, prompt_token_ids=prompt_token_ids, arrival_time=arrival_time)
        
        while True:
            if request_id not in self.request_events:
                return

            if not self.is_engine_running:
                await self.engine_step(request_id)

            try:
                await asyncio.wait_for(request_event.wait(),
                                        timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            request_event.clear()

            request_event = self.request_outputs[request_id]
            yield request_output

            if request_output.finished:
                if self.log_requests:
                    logger.info(f"Finished request {request_id}.")

                del self.request_outputs[request_event]
                del self.request_events[request_id]

                if not self.is_engine_running:
                    await self.engine_step()
                break

    async def abort(self, request_id: str) -> None:

        if request_id not in self.request_outputs:
            return

        if self.log_requests:
            logger.info(f"Aborted requests {request_id}.")

        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_id)
        else:
            self.engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.self.kicking_request_id = None

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> "AsyncAphrodite":
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        distributed_init_method, devices = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        engine = cls(engine_args.worker_use_ray, engine_args.engine_use_ray, not engine_args.disable_log_requests, *engine_configs, distributed_init_method, devices, log_stats=not engine_args.disable_log_stats)
        return engine