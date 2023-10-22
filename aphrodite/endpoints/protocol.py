from typing import List, Optional, Union
from pydantic import BaseModel, Field, root_validator, conint, confloat, conlist, NonNegativeFloat, NonNegativeInt, PositiveInt

class SamplingParams(BaseModel):
    n: int = Field(1, alias="n")
    best_of: Optional[int] = Field(None, alias="best_of")
    presence_penalty: float = Field(0.0, alias="presence_penalty")
    frequency_penalty: float = Field(0.0, alias="rep_pen")
    temperature: float = Field(1.0, alias="temperature")
    top_p: float = Field(1.0, alias="top_p")
    top_k: float = Field(-1, alias="top_k")
    tfs: float = Field(1.0, alias="tfs")
    eta_cutoff: float = Field(0.0, alias="eta_cutoff")
    epsilon_cutoff: float = Field(0.0, alias="epsilon_cutoff")
    typical_p: float = Field(1.0, alias="typical_p")
    use_beam_search: bool = Field(False, alias="use_beam_search")
    length_penalty: float = Field(1.0, alias="length_penalty")
    early_stopping: Union[bool, str] = Field(False, alias="early_stopping")
    stop: Union[None, str, List[str]] = Field(None, alias="stop_sequence")
    ignore_eos: bool = Field(False, alias="ignore_eos")
    max_tokens: int = Field(16, alias="max_length")
    logprobs: Optional[int] = Field(None, alias="logprobs")

    @root_validator
    def validate_best_of(cls, values):
        best_of = values.get("best_of")
        n = values.get("n")
        if best_of is not None and (best_of <= 0 or best_of > n):
            raise ValueError("best_of must be a positive integer less than or equal to n")
        return values

class KAIGenerationInputSchema(BaseModel):
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
    tfs: Optional[confloat(ge=0, le=1)] = 1.0
    eps_cutoff: Optional[confloat(ge=0,le=1000)] = 0.0
    eta_cutoff: Optional[NonNegativeFloat] = 0.0
    typical: Optional[confloat(ge=0, le=1)] = 1.0
    temperature: Optional[NonNegativeFloat] = 1.0
    use_memory: Optional[bool]
    use_story: Optional[bool]
    use_authors_note: Optional[bool]
    use_world_info: Optional[bool]
    use_userscripts: Optional[bool]
    soft_prompt: Optional[str]
    disable_output_formatting: Optional[bool]
    frmtrmblln: Optional[bool]
    frmtrmspch: Optional[bool]
    singleline: Optional[bool]
    use_default_badwordsids: Optional[bool]
    disable_input_formatting: Optional[bool]
    frmtadsnsp: Optional[bool]
    quiet: Optional[bool]
    sampler_order: Optional[conlist(int, min_items=6)]
    sampler_seed: Optional[conint(ge=0, le=2**64 - 1)]
    sampler_full_determinism: Optional[bool]
    stop_sequence: Optional[List[str]]

    @root_validator
    def check_context(cls, values):
        assert values.get("max_length") <= values.get("max_context_length"), f"max_length must not be larger than max_context_length"
        return values