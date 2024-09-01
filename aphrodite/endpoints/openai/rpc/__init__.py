from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from aphrodite.common.sampling_params import SamplingParams
from aphrodite.inputs import PromptInputs
from aphrodite.lora.request import LoRARequest
from aphrodite.prompt_adapter.request import PromptAdapterRequest

APHRODITE_RPC_SUCCESS_STR = "SUCCESS"
APHRODITE_RPC_HEALTHY_STR = "HEALTHY"


@dataclass
class RPCGenerateRequest:
    inputs: PromptInputs
    sampling_params: SamplingParams
    request_id: str
    lora_request: Optional[LoRARequest] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None


@dataclass
class RPCAbortRequest:
    request_id: str


class RPCUtilityRequest(Enum):
    IS_SERVER_READY = 1
    GET_MODEL_CONFIG = 2
    GET_DECODING_CONFIG = 3
    GET_PARALLEL_CONFIG = 4
    GET_SCHEDULER_CONFIG = 5
    GET_LORA_CONFIG = 6
    DO_LOG_STATS = 7
    CHECK_HEALTH = 8


RPC_REQUEST_TYPE = Union[RPCGenerateRequest, RPCAbortRequest,
                         RPCUtilityRequest]
