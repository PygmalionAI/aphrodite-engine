"""Sampling parameters for text generation."""
import copy
from enum import IntEnum
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pydantic import conint

_SAMPLING_EPS = 1e-5


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2
    BEAM = 3


LogitsProcessorFunc = Callable[[torch.Tensor, List[List[int]]], None]
"""LogitsProcessorFunc takes a logits tensor and corresponding lists of
previously generated output tokens, and modifies the logits tensor."""


class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support multiple additional samplers which are not supported
    by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far.
            freq_pen is applied additively while
            rep_pen is applied multiplicatively.
            Must be in [1, inf). Set to 1 to disable the effect.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        top_a: Float that controls the cutoff for Top-A sampling.
            Exact cutoff is top_a*max_prob**2. Must be in [0,inf], 0 to disable.
        min_p: Float that controls the cutoff for min-p sampling.
            Exact cutoff is min_p*max_prob. Must be in [0,1], 0 to disable.
        tfs: Float that controls the cumulative approximate curvature of the
            distribution to retain for Tail Free Sampling.
            Must be in (0, 1]. Set to 1 to disable
        eta_cutoff: Float that controls the cutoff threshold for Eta sampling
            (a form of entropy adaptive truncation sampling)
            threshold is computed as min(eta, sqrt(eta)*entropy(probs)).
            Specified in units of 1e-4. Set to 0 to disable
        epsilon_cutoff: Float that controls the cutoff threshold for
            Epsilon sampling (simple probability threshold truncation).
            Specified in units of 1e-4. Set to 0 to disable.
        typical_p: Float that controls the cumulative probability of tokens
            closest in surprise to the expected surprise to consider.
            Must be in (0, 1]. Set to 1 to disable.
        mirostat_mode: Can either be 0 (disabled) or 2 (Mirostat v2).
        mirostat_tau: Target "surprisal" that mirostat works towards.
            Range [0, inf).
        mirostat_eta: Rate at which mirostat updates its internal surprisal
            value. Range [0, inf).
        dynatemp_min: Minimum temperature for dynatemp sampling.
            Range [0, inf).
        dynatemp_max: Maximum temperature for dynatemp sampling.
            Range [0, inf).
        dynatemp_exponent: Exponent for dynatemp sampling. Range [0, inf).
        smoothing_factor: Smoothing factor for Quadratic Sampling.
        smoothing_curve: Smoothing curve for Quadratic (Cubic) Sampling.
        seed: Random seed to use for the generation.
        use_beam_search: Whether to use beam search instead of sampling.
        length_penalty: Float that penalizes sequences based on their length.
            Used in beam search.
        early_stopping: Controls the stopping condition for beam search. It
            accepts the following values: `True`, where the generation stops as
            soon as there are `best_of` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very
            unlikely to find better candidates; `"never"`, where the beam search
            procedure only stops when there cannot be better candidates
            (canonical beam search algorithm).
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        include_stop_str_in_output: Whether to include the stop strings in
            output text. Defaults to False.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens: Minimum number of tokens to generate per output sequence
            before EOS or stop tokens are generated.
        logprobs: Number of log probabilities to return per output token.
            Note that the implementation follows the OpenAI API: The return
            result includes the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens. The API will always return the
            log probability of the sampled token, so there  may be up to
            `logprobs+1` elements in the response.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        detokenize: Whether to detokenize the output. Defaults to True.
        custom_token_bans: List of token IDs to ban from generating
        skip_special_tokens: Whether to skip special tokens in the output.
            defaults to true.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens in the output. Defaults to True.
        logits_processors: List of LogitsProcessors to change the probability
            of token prediction at runtime.
        truncate_prompt_tokens: If set to an integer k, will use only the last
            k tokens from the prompt (i.e. left-truncation). Defaults to None
            (i.e. no truncation).
    """

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        top_a: float = 0.0,
        min_p: float = 0.0,
        tfs: float = 1.0,
        eta_cutoff: float = 0.0,
        epsilon_cutoff: float = 0.0,
        typical_p: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 0,
        mirostat_eta: float = 0,
        dynatemp_min: float = 0,
        dynatemp_max: float = 0,
        dynatemp_exponent: float = 1,
        smoothing_factor: float = 0.0,
        smoothing_curve: float = 1.0,
        seed: Optional[int] = None,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        stop: Union[None, str, List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: Optional[int] = 16,
        min_tokens: int = 0,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        detokenize: bool = True,
        custom_token_bans: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        logits_processors: Optional[List[LogitsProcessorFunc]] = None,
        truncate_prompt_tokens: Optional[conint(ge=1)] = None,
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.top_a = top_a
        self.min_p = min_p
        self.tfs = tfs
        self.eta_cutoff = eta_cutoff
        self.epsilon_cutoff = epsilon_cutoff
        self.typical_p = typical_p
        self.mirostat_mode = mirostat_mode
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.dynatemp_min = dynatemp_min
        self.dynatemp_max = dynatemp_max
        self.dynatemp_exponent = dynatemp_exponent
        self.smoothing_factor = smoothing_factor
        self.smoothing_curve = smoothing_curve
        self.seed = seed
        self.use_beam_search = use_beam_search
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        if stop is None:
            self.stop = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)
        self.stop_token_ids = stop_token_ids or []
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs
        # NOTE: This parameter is only exposed at the engine level for now.
        # It is not exposed in the OpenAI API server, as the OpenAI API does
        # not support returning only a list of token IDs.
        self.detokenize = detokenize
        self.custom_token_bans = custom_token_bans or []
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.logits_processors = logits_processors or []
        self.include_stop_str_in_output = include_stop_str_in_output
        self.truncate_prompt_tokens = truncate_prompt_tokens

        self.default_values = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "top_a": 0.0,
            "min_p": 0.0,
            "tfs": 1.0,
            "eta_cutoff": 0.0,
            "epsilon_cutoff": 0.0,
            "typical_p": 1.0,
            "mirostat_mode": 0,
            "mirostat_tau": 0,
            "mirostat_eta": 0,
            "dynatemp_min": 0,
            "dynatemp_max": 0,
            "dynatemp_exponent": 1,
            "smoothing_factor": 0.0,
            "smoothing_curve": 1.0,
            "seed": None,
            "use_beam_search": False,
            "length_penalty": 1.0,
            "early_stopping": False,
            "stop": [],
            "stop_token_ids": [],
            "ignore_eos": False,
            "max_tokens": 16,
            "min_tokens": 0,
            "logprobs": None,
            "prompt_logprobs": None,
            "detokenize": True,
            "custom_token_bans": [],
            "skip_special_tokens": True,
            "spaces_between_special_tokens": True,
            "include_stop_str_in_output": False,
            "truncate_prompt_tokens": None,
        }

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.output_text_buffer_length = 0

        self._verify_args()
        if self.use_beam_search:
            self._verify_beam_search()
        else:
            self._verify_non_beam_search()
            if self.temperature < _SAMPLING_EPS:
                # Zero temperature means greedy sampling.
                self.top_p = 1.0
                self.top_k = -1
                self.min_p = 0.0
                self.top_a = 0.0
                self._verify_greedy_sampling()
        # injected by the engine
        self.eos_token_id = None

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of < self.n:
            raise ValueError(f"best_of must be greater than or equal to n, "
                             f"got n={self.n} and best_of={self.best_of}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be in [1, inf), got "
                             f"{self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if self.top_a < 0:
            raise ValueError(f"top_a must be non negative, got {self.top_a}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if not 0.0 < self.tfs <= 1.0:
            raise ValueError(f"tfs must be in (0, 1], got {self.tfs}.")
        if self.epsilon_cutoff < 0.0 or self.epsilon_cutoff > 1000.0:
            raise ValueError("epsilon_cutoff must be in [0, 1000], got "
                             f"{self.epsilon_cutoff}.")
        # pylint: disable=unneeded-not
        if not self.eta_cutoff >= 0:
            raise ValueError(
                f"eta_cutoff must be non negative, got {self.eta_cutoff}.")
        if not 0.0 <= self.typical_p <= 1.0:
            raise ValueError(
                f"typical_p must be in (0, 1], got {self.typical_p}.")
        if not self.dynatemp_min >= 0:
            raise ValueError(
                f"dynatemp_min must be non negative, got {self.dynatemp_min}.")
        if not self.dynatemp_max >= 0:
            raise ValueError(
                f"dynatemp_max must be non negative, got {self.dynatemp_max}.")
        if not self.dynatemp_exponent >= 0:
            raise ValueError(f"dynatemp_exponent must be non negative, got "
                             f"{self.dynatemp_exponent}.")
        if not self.smoothing_factor >= 0:
            raise ValueError(f"smoothing_factor must be non negative, got "
                             f"{self.smoothing_factor}.")
        if not self.smoothing_curve >= 1.0:
            raise ValueError(f"smoothing_curve must larger than 1, got "
                             f"{self.smoothing_curve}.")
        if self.mirostat_mode:
            if not self.mirostat_mode == 2:
                raise ValueError(
                    "Only Mirostat v2 (2) and disabled (0) supported, "
                    f"got {self.mirostat_mode}")
            if not self.mirostat_eta >= 0:
                raise ValueError(
                    f"mirostat_eta must be positive, got {self.mirostat_eta}")
            if not self.mirostat_tau >= 0:
                raise ValueError(
                    f"mirostat_tau must be positive, got {self.mirostat_tau}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, "
                             f"got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError("prompt_logprobs must be non-negative, got "
                             f"{self.prompt_logprobs}.")
        if (self.truncate_prompt_tokens is not None
                and self.truncate_prompt_tokens < 1):
            raise ValueError(f"truncate_prompt_tokens must be >= 1, "
                             f"got {self.truncate_prompt_tokens}")
        if any(not stop_str for stop_str in self.stop):
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop.")

    def _verify_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError("best_of must be greater than 1 when using beam "
                             f"search. Got {self.best_of}.")
        if self.temperature > _SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")
        if self.early_stopping not in [True, False, "never"]:
            raise ValueError(
                f"early_stopping must be True, False, or 'never', "
                f"got {self.early_stopping}.")

    def _verify_non_beam_search(self) -> None:
        if self.early_stopping is not False:
            raise ValueError("early_stopping is not effective and must be "
                             "False when not using beam search.")
        if (self.length_penalty < 1.0 - _SAMPLING_EPS
                or self.length_penalty > 1.0 + _SAMPLING_EPS):
            raise ValueError(
                "length_penalty is not effective and must be the "
                "default value of 1.0 when not using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError("best_of must be 1 when using greedy sampling."
                             f"Got {self.best_of}.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

    def update_from_generation_config(
            self, generation_config: Dict[str, Any]) -> None:
        """Update if there are non-default values from generation_config"""
        # Update eos_token_id for generation
        if eos_ids := generation_config.get("eos_token_id"):
            # it can be either int or list of int
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            original_stop_token_ids = set(self.stop_token_ids)
            original_stop_token_ids.update(eos_ids)
            self.stop_token_ids = list(original_stop_token_ids)

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.use_beam_search:
            return SamplingType.BEAM
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    def clone(self) -> "SamplingParams":
        """Deep copy excluding LogitsProcessor objects.
        LogitsProcessor objects are excluded because they may contain an
        arbitrary, nontrivial amount of data.
        """

        logit_processor_refs = None if self.logits_processors is None else {
            id(lp): lp
            for lp in self.logits_processors
        }
        return copy.deepcopy(self, memo=logit_processor_refs)

    def __repr__(self) -> str:
        repr_str = "SamplingParams("
        for param, default_value in self.default_values.items():
            current_value = getattr(self, param)
            if current_value != default_value:
                repr_str += f"{param}={current_value}, "
        repr_str = repr_str.rstrip(', ') + ")"
        return repr_str
