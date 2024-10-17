from typing import List, Optional, Union

from pydantic import BaseModel, Field, root_validator


class SamplingParams(BaseModel):
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
