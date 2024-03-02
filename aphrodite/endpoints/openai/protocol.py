# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel, Field, model_validator, conint, confloat,
    NonNegativeFloat, NonNegativeInt, PositiveInt)
import torch

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


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    include_stop_str_in_output: Optional[bool] = False
    stream: Optional[bool] = False
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
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
    prompt_logprobs: Optional[int] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    custom_token_bans: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    add_generation_prompt: Optional[bool] = True
    echo: Optional[bool] = False
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None

    def to_sampling_params(self) -> SamplingParams:
        if self.logprobs and not self.top_logprobs:
            raise ValueError("Top logprobs must be set when logprobs is.")
        
        logits_processors = None
        if self.logit_bias:

            def logit_bias_logits_processor(
                    token_ids: List[int],
                    logits: torch.Tensor) -> torch.Tensor:
                for token_id, bias in self.logit_bias.items():
                    # Clamp the bias between -100 and 100 per OpenAI API spec
                    bias = min(100, max(-100, bias))
                    logits[int(token_id)] += bias
                return logits

            logits_processors = [logit_bias_logits_processor]
                    
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens,
            logprobs=self.top_logprobs if self.logprobs else None,
            prompt_logprobs=self.top_logprobs if self.echo else None,
            temperature=self.temperature,
            top_p=self.top_p,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
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
            stop=self.stop,
            best_of=self.best_of,
            include_stop_str_in_output=self.include_stop_str_in_output,
            seed=self.seed,
            logits_processors=logits_processors,
        )
    
    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data


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
    seed: Optional[int] = None
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
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None

    def to_sampling_params(self) -> SamplingParams:
        echo_without_generation = self.echo and self.max_tokens == 0

        logits_processors = None
        if self.logit_bias:

            def logit_bias_logits_processor(
                    token_ids: List[int],
                    logits: torch.Tensor) -> torch.Tensor:
                for token_id, bias in self.logit_bias.items():
                    bias = min(100, max(-100, bias))
                    logits[int(token_id)] += bias
                return logits

            logits_processors = [logit_bias_logits_processor]
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens if not echo_without_generation else 1,
            temperature=self.temperature,
            top_p=self.top_p,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
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
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs if self.echo else None,
            stop_token_ids=self.stop_token_ids,
            custom_token_bans=self.custom_token_bans,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            stop=self.stop,
            best_of=self.best_of,
            include_stop_str_in_output=self.include_stop_str_in_output,
            seed=self.seed,
            logits_processors=logits_processors,
        )
    
    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data


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
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
    logprobs: Optional[LogProbs] = None


class Prompt(BaseModel):
    prompt: str

# ========== Kobold API ========== #
class SamplingParamsKobold(BaseModel):
    n: int = Field(1, alias="n")
    best_of: Optional[int] = Field(None, alias="best_of")
    presence_penalty: float = Field(0.0, alias="presence_penalty")
    frequency_penalty: float = Field(0.0, alias="rep_pen")
    temperature: float = Field(1.0, alias="temperature")
    dynatemp_range: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    top_p: float = Field(1.0, alias="top_p")
    top_k: float = Field(-1, alias="top_k")
    min_p: float = Field(0.0, alias="min_p")
    top_a: float = Field(0.0, alias="top_a")
    tfs: float = Field(1.0, alias="tfs")
    eta_cutoff: float = Field(0.0, alias="eta_cutoff")
    epsilon_cutoff: float = Field(0.0, alias="epsilon_cutoff")
    typical_p: float = Field(1.0, alias="typical_p")
    use_beam_search: bool = Field(False, alias="use_beam_search")
    length_penalty: float = Field(1.0, alias="length_penalty")
    early_stopping: Union[bool, str] = Field(False, alias="early_stopping")
    stop: Union[None, str, List[str]] = Field(None, alias="stop_sequence")
    include_stop_str_in_output: Optional[bool] = False
    ignore_eos: bool = Field(False, alias="ignore_eos")
    max_tokens: int = Field(16, alias="max_length")
    logprobs: Optional[int] = Field(None, alias="logprobs")
    custom_token_bans: Optional[List[int]] = Field(None,
                                                   alias="custom_token_bans")


class KAIGenerationInputSchema(BaseModel):
    genkey: Optional[str]
    prompt: str
    n: Optional[conint(ge=1, le=5)] = 1
    max_context_length: PositiveInt
    max_length: PositiveInt
    rep_pen: Optional[confloat(ge=1)] = 1.0
    rep_pen_range: Optional[NonNegativeInt]
    rep_pen_slope: Optional[NonNegativeFloat]
    top_k: Optional[NonNegativeInt] = 0.0
    top_a: Optional[NonNegativeFloat] = 0.0
    top_p: Optional[confloat(ge=0, le=1)] = 1.0
    min_p: Optional[confloat(ge=0, le=1)] = 0.0
    tfs: Optional[confloat(ge=0, le=1)] = 1.0
    eps_cutoff: Optional[confloat(ge=0, le=1000)] = 0.0
    eta_cutoff: Optional[NonNegativeFloat] = 0.0
    typical: Optional[confloat(ge=0, le=1)] = 1.0
    temperature: Optional[NonNegativeFloat] = 1.0
    dynatemp_range: Optional[NonNegativeFloat] = 0.0
    dynatemp_exponent: Optional[NonNegativeFloat] = 1.0
    smoothing_factor: Optional[NonNegativeFloat] = 0.0
    # use_memory: Optional[bool]
    # use_story: Optional[bool]
    # use_authors_note: Optional[bool]
    # use_world_info: Optional[bool]
    # use_userscripts: Optional[bool]
    # soft_prompt: Optional[str]
    # disable_output_formatting: Optional[bool]
    # frmtrmblln: Optional[bool]
    # frmtrmspch: Optional[bool]
    # singleline: Optional[bool]
    use_default_badwordsids: Optional[bool]
    mirostat: Optional[int] = 0
    mirostat_tau: Optional[float] = 0.0
    mirostat_eta: Optional[float] = 0.0
    # disable_input_formatting: Optional[bool]
    # frmtadsnsp: Optional[bool]
    # quiet: Optional[bool]
    # pylint: disable=unexpected-keyword-arg
    # sampler_order: Optional[conlist(int)]
    # sampler_seed: Optional[conint(ge=0, le=2**64 - 1)]
    # sampler_full_determinism: Optional[bool]
    stop_sequence: Optional[List[str]]
    include_stop_str_in_output: Optional[bool] = False
