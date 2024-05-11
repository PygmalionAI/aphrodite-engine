# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import (AliasChoices, BaseModel, Field, conint, model_validator,
                      root_validator)

from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.utils import random_uuid
from aphrodite.common.logits_processor import BiasLogitsProcessor


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
    is_blocking: bool = False


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


class ResponseFormat(BaseModel):
    # type must be "json_object" or "text"
    type: str = Literal["text", "json_object"]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = 0
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
    dynatemp_min: Optional[float] = 0.0
    dynatemp_max: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
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
    guided_grammar: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
    guided_decoding_backend: Optional[str] = Field(
        default="outlines",
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))

    def to_sampling_params(self, vocab_size: int) -> SamplingParams:
        if self.logprobs and not self.top_logprobs:
            raise ValueError("Top logprobs must be set when logprobs is.")
        if self.top_k == 0:
            self.top_k = -1

        logits_processors = []
        if self.logit_bias:
            biases = {
                int(tok): max(-100, min(float(bias), 100))
                for tok, bias in self.logit_bias.items()
                if 0 < int(tok) < vocab_size
            }
            logits_processors.append(BiasLogitsProcessor(biases))

        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
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
            dynatemp_min=self.dynatemp_min,
            dynatemp_max=self.dynatemp_max,
            dynatemp_exponent=self.dynatemp_exponent,
            smoothing_factor=self.smoothing_factor,
            smoothing_curve=self.smoothing_curve,
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
    min_tokens: Optional[int] = 0
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
    dynatemp_min: Optional[float] = Field(0.0,
                                          validation_alias=AliasChoices(
                                              "dynatemp_min", "dynatemp_low"),
                                          description="Aliases: dynatemp_low")
    dynatemp_max: Optional[float] = Field(0.0,
                                          validation_alias=AliasChoices(
                                              "dynatemp_max", "dynatemp_high"),
                                          description="Aliases: dynatemp_high")
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    custom_token_bans: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    truncate_prompt_tokens: Optional[conint(ge=1)] = None
    grammar: Optional[str] = None
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
    guided_decoding_backend: Optional[str] = Field(
        default="outlines",
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))

    def to_sampling_params(self, vocab_size: int) -> SamplingParams:
        echo_without_generation = self.echo and self.max_tokens == 0
        if self.top_k == 0:
            self.top_k = -1

        logits_processors = []
        if self.logit_bias:
            biases = {
                int(tok): max(-100, min(float(bias), 100))
                for tok, bias in self.logit_bias.items()
                if 0 < int(tok) < vocab_size
            }
            logits_processors.append(BiasLogitsProcessor(biases))

        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
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
            dynatemp_min=self.dynatemp_min,
            dynatemp_max=self.dynatemp_max,
            dynatemp_exponent=self.dynatemp_exponent,
            smoothing_factor=self.smoothing_factor,
            smoothing_curve=self.smoothing_curve,
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
            truncate_prompt_tokens=self.truncate_prompt_tokens,
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
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None
    stop_reason: Union[None, int, str] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


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
    stop_reason: Union[None, int, str] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


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
    stop_reason: Union[None, int, str] = None


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
    stop_reason: Union[None, int, str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
    logprobs: Optional[LogProbs] = None


class EmbeddingsRequest(BaseModel):
    input: List[str] = Field(
        ..., description="List of input texts to generate embeddings for.")
    encoding_format: str = Field(
        "float",
        description="Encoding format for the embeddings. "
        "Can be 'float' or 'base64'.")
    model: Optional[str] = Field(
        None,
        description="Name of the embedding model to use. "
        "If not provided, the default model will be used.")


class EmbeddingObject(BaseModel):
    object: str = Field("embedding", description="Type of the object.")
    embedding: List[float] = Field(
        ..., description="Embedding values as a list of floats.")
    index: int = Field(
        ...,
        description="Index of the input text corresponding to "
        "the embedding.")


class EmbeddingsResponse(BaseModel):
    object: str = Field("list", description="Type of the response object.")
    data: List[EmbeddingObject] = Field(
        ..., description="List of embedding objects.")
    model: str = Field(..., description="Name of the embedding model used.")
    usage: UsageInfo = Field(..., description="Information about token usage.")


class Prompt(BaseModel):
    prompt: str


# ========== KoboldAI ========== #


class KoboldSamplingParams(BaseModel):
    n: int = Field(1, alias="n")
    best_of: Optional[int] = Field(None, alias="best_of")
    presence_penalty: float = Field(0.0, alias="presence_penalty")
    frequency_penalty: float = Field(0.0, alias="rep_pen")
    temperature: float = Field(1.0, alias="temperature")
    dynatemp_range: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
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

    @root_validator(pre=False, skip_on_failure=True)
    def validate_best_of(cls, values):  # pylint: disable=no-self-argument
        best_of = values.get("best_of")
        n = values.get("n")
        if best_of is not None and (best_of <= 0 or best_of > n):
            raise ValueError(
                "best_of must be a positive integer less than or equal to n")
        return values


class KAIGenerationInputSchema(BaseModel):
    genkey: Optional[str] = None
    prompt: str
    n: Optional[int] = 1
    max_context_length: int
    max_length: int
    rep_pen: Optional[float] = 1.0
    rep_pen_range: Optional[int] = None
    rep_pen_slope: Optional[float] = None
    top_k: Optional[int] = 0
    top_a: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    tfs: Optional[float] = 1.0
    eps_cutoff: Optional[float] = 0.0
    eta_cutoff: Optional[float] = 0.0
    typical: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    dynatemp_range: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
    use_memory: Optional[bool] = None
    use_story: Optional[bool] = None
    use_authors_note: Optional[bool] = None
    use_world_info: Optional[bool] = None
    use_userscripts: Optional[bool] = None
    soft_prompt: Optional[str] = None
    disable_output_formatting: Optional[bool] = None
    frmtrmblln: Optional[bool] = None
    frmtrmspch: Optional[bool] = None
    singleline: Optional[bool] = None
    use_default_badwordsids: Optional[bool] = None
    mirostat: Optional[int] = 0
    mirostat_tau: Optional[float] = 0.0
    mirostat_eta: Optional[float] = 0.0
    disable_input_formatting: Optional[bool] = None
    frmtadsnsp: Optional[bool] = None
    quiet: Optional[bool] = None
    # pylint: disable=unexpected-keyword-arg
    sampler_order: Optional[Union[List, str]] = Field(default_factory=list)
    sampler_seed: Optional[int] = None
    sampler_full_determinism: Optional[bool] = None
    stop_sequence: Optional[List[str]] = None
    include_stop_str_in_output: Optional[bool] = False

    @root_validator(pre=False, skip_on_failure=True)
    def check_context(cls, values):  # pylint: disable=no-self-argument
        assert values.get("max_length") <= values.get(
            "max_context_length"
        ), "max_length must not be larger than max_context_length"
        return values
