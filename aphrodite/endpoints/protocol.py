from typing import List, Optional, Union
from pydantic import BaseModel, Field, root_validator, conint, confloat, conlist, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

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
    rep_pen: Optional[confloat(ge=1)] = Field(1.0, description="Base repetition penalty value.")
    rep_pen_range: Optional[NonNegativeInt] = Field(description="Repetition penalty range.")
    rep_pen_slope: Optional[NonNegativeFloat] = Field(description="Repetition penalty slope.")
    top_k: Optional[NonNegativeInt] = Field(0.0, description="Top-k sampling value.")
    top_a: Optional[NonNegativeFloat] = Field(0.0, description="Top-a sampling value.")
    top_p: Optional[confloat(ge=0, le=1)] = Field(1.0, description="Top-p sampling value.")
    tfs: Optional[confloat(ge=0, le=1)] = Field(1.0, description="Tail free sampling value.")
    typical: Optional[confloat(ge=0, le=1)] = Field(1.0, description="Typical sampling value.")
    temperature: Optional[PositiveFloat] = Field(1.0, description="Temperature value.")
    prompt: str = Field(description="This is the submission.")
    use_memory: Optional[bool] = Field(None, description="Whether or not to use the memory from the KoboldAI GUI when generating text.")
    use_story: Optional[bool] = Field(None, description="Whether or not to use the story from the KoboldAI GUI when generating text.")
    use_authors_note: Optional[bool] = Field(None, description="Whether or not to use the author's note from the KoboldAI GUI when generating text. This has no effect unless `use_story` is also enabled.")
    use_world_info: Optional[bool] = Field(None, description="Whether or not to use the world info from the KoboldAI GUI when generating text.")
    use_userscripts: Optional[bool] = Field(None, description="Whether or not to use the userscripts from the KoboldAI GUI when generating text.")
    soft_prompt: Optional[str] = Field(None, description="Soft prompt to use when generating. If set to the empty string or any other string containing no non-whitespace characters, uses no soft prompt.")
    max_length: PositiveInt = Field(description="Number of tokens to generate.")
    max_context_length: PositiveInt = Field(description="Maximum number of tokens to send to the model.")
    n: Optional[conint(ge=1, le=5)] = Field(1, description="Number of outputs to generate.")
    disable_output_formatting: bool = Field(False, description="When enabled, all output formatting options default to `false` instead of the value in the KoboldAI GUI.")
    frmttriminc: Optional[bool] = Field(description="Output formatting option. When enabled, removes some characters from the end of the output such that the output doesn't end in the middle of a sentence. If the output is less than one sentence long, does nothing.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI.")
    frmtrmblln: Optional[bool] = Field(description="Output formatting option. When enabled, replaces all occurrences of two or more consecutive newlines in the output with one newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI.")
    frmtrmspch: Optional[bool] = Field(description="Output formatting option. When enabled, removes `#/@%{}+=~|\^<>` from the output.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI.")
    singleline: Optional[bool] = Field(description="Output formatting option. When enabled, removes everything after the first line of the output, including the newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI.")
    use_default_badwordsids: Optional[bool] = Field(True, description="Ban tokens that commonly worsen the writing experience for continuous story writing")
    disable_input_formatting: Optional[bool] = Field(description="When enabled, all input formatting options default to `false` instead of the value in the KoboldAI GUI")
    frmtadsnsp: Optional[bool] = Field(description="Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action.\n\nIf `disable_input_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI.")
    quiet: Optional[bool] = Field(description="When enabled, Generated output will not be displayed in the console.")
    sampler_order: Optional[conlist(int, min_items=6)] = Field(description="Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers.")
    sampler_seed: Optional[conint(ge=0, le=2**64 - 1)] = Field(description="RNG seed to use for sampling. If not specified, the global RNG will be used.")
    sampler_full_determinism: Optional[bool] = Field(description="If enabled, the generated text will always be the same as long as you use the same RNG seed, input and settings. If disabled, only the *sequence* of generated texts that you get when repeatedly generating text will be the same given the same RNG seed, input and settings.")
    stop_sequence: Optional[List[str]] = Field(description="An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence.")

    @root_validator
    def check_context(cls, values):
        assert values.get("max_length") <= values.get("max_context_length"), f"max_length must not be larger than max_context_length"
        return values