from typing import AsyncGenerator, List, Optional, Protocol, runtime_checkable

from transformers import PreTrainedTokenizer

from aphrodite.common.config import DecodingConfig, ModelConfig
from aphrodite.common.outputs import EmbeddingRequestOutput, RequestOutput
from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import SamplerOutput
from aphrodite.inputs.data import PromptInputs
from aphrodite.lora.request import LoRARequest
from aphrodite.processing.scheduler import SchedulerOutputs
from aphrodite.prompt_adapter.request import PromptAdapterRequest


@runtime_checkable
class AsyncEngineClient(Protocol):
    """Protocol class for Clients to AsyncAphrodite"""

    @property
    def is_running(self) -> bool:
        ...

    @property
    def is_stopped(self) -> bool:
        ...

    @property
    def errored(self) -> bool:
        ...

    def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generates outputs for a request"""
        ...

    def encode(
        self,
        inputs: PromptInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """Generate outputs for a request from an embedding model."""
        ...

    async def abort(self, request_id: str) -> None:
        """Abort a request.
        Args:
            request_id: The unique id of the request.
        """
        ...

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the Aphrodite engine."""
        ...

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the Aphrodite engine."""
        ...

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> PreTrainedTokenizer:
        """Get the appropriate Tokenizer for the request"""
        ...

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        pass

    async def check_health(self) -> None:
        """Raise if unhealthy"""
