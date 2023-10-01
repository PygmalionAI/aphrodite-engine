from typing import List, Optional, Union
from pydantic import BaseModel, Field, root_validator

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