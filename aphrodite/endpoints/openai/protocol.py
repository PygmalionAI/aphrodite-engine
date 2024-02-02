# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from aphrodite.common.utils import random_uuid
from aphrodite.common.sampling_params import SamplingParams


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pygmalionai"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class Function(BaseModel):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str
    function: Function


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Optional[Any] = None
    # See : https://json-schema.org/understanding-json-schema/reference/object


class ChatCompletionToolParam(BaseModel):
    type: str = "function"
    function: FunctionDefinition = None


class ChatCompletionSystemMessage(BaseModel):
    role: Literal["system"]
    content: str
    name: Optional[str] = None


class ChatCompletionUserMessage(BaseModel):
    role: Literal["user"]
    content: Union[str, List[str]]
    name: Optional[str] = None


class ChatCompletionAssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class ChatCompletionToolMessage(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Union[ChatCompletionToolMessage,
                         ChatCompletionAssistantMessage,
                         ChatCompletionUserMessage,
                         ChatCompletionSystemMessage]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    include_stop_str_in_output: Optional[bool] = False
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    top_a: Optional[float] = 0.0
    min_p: Optional[float] = 0.0
    mirostat_mode: Optional[int] = 0
    mirostat_tau: Optional[float] = 0.0
    mirostat_eta: Optional[float] = 0.0
    dynatemp_range: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    custom_token_bans: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    add_generation_prompt: Optional[bool] = True
    echo: Optional[bool] = False
    tools: Optional[List[ChatCompletionToolParam]] = None
    tool_choice: Optional[str] = None

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            best_of=self.best_of,
            top_k=self.top_k,
            top_a=self.top_a,
            min_p=self.min_p,
            mirostat_mode=self.mirostat_mode,
            mirostat_tau=self.mirostat_tau,
            mirostat_eta=self.mirostat_eta,
            dynatemp_range=self.dynatemp_range,
            dynatemp_exponent=self.dynatemp_exponent,
            smoothing_factor=self.smoothing_factor,
            ignore_eos=self.ignore_eos,
            use_beam_search=self.use_beam_search,
            stop_token_ids=self.stop_token_ids,
            custom_token_bans=self.custom_token_bans,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output
        )


class CompletionRequest(BaseModel):
    model: str
    # a string, array of strings, array of tokens, or array of token arrays
    prompt: Union[List[int], List[List[int]], str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    include_stop_str_in_output: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    top_k: Optional[int] = -1
    top_a: Optional[float] = 0.0
    min_p: Optional[float] = 0.0
    mirostat_mode: Optional[int] = 0
    mirostat_tau: Optional[float] = 0.0
    mirostat_eta: Optional[float] = 0.0
    dynatemp_range: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    custom_token_bans: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    grammar: Optional[str] = None

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            best_of=self.best_of,
            top_k=self.top_k,
            top_a=self.top_a,
            min_p=self.min_p,
            mirostat_mode=self.mirostat_mode,
            mirostat_tau=self.mirostat_tau,
            mirostat_eta=self.mirostat_eta,
            dynatemp_range=self.dynatemp_range,
            dynatemp_exponent=self.dynatemp_exponent,
            smoothing_factor=self.smoothing_factor,
            ignore_eos=self.ignore_eos,
            use_beam_search=self.use_beam_search,
            stop_token_ids=self.stop_token_ids,
            custom_token_bans=self.custom_token_bans,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            logprobs=self.logprobs,
            prompt_logprobs=self.logprobs if self.echo else None,
            logits_processors=self.grammar,
        )


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: str
    type: str
    function: Function


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

class Prompt(BaseModel):
    prompt: str