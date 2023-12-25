from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import SequenceData
from aphrodite.common.utils import in_wsl

_SAMPLING_EPS = 1e-5


class PersistentMetadata:

    def __init__(self, metadata: Optional[Dict[int, dict]] = None):
        self._metadata: Dict[int, dict] = metadata or {}

    def get(self, seq_id: int) -> dict:
        return self._metadata.get(seq_id, {})


class OutputMetadata(PersistentMetadata):

    def add(self, seq_id: int, key, val) -> None:
        if seq_id not in self._metadata:
            self._metadata[seq_id] = {}
        self._metadata[seq_id][key] = val


class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.
    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        selected_token_indices: Token indices selected for sampling.
        categorized_sample_indices: SamplingType -> token indicies to sample.
        persistent_metadata: Metadata that persists across iterations.
        output_metadata: the metadata of the output.
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Dict[SamplingType, torch.Tensor],
        persistent_metadata: Optional[PersistentMetadata] = None,
        output_metadata: Optional[OutputMetadata] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.persistent_metadata = persistent_metadata or PersistentMetadata()
        self.output_metadata = output_metadata or OutputMetadata()

        self.num_prompts = len(prompt_lens)

    def __repr__(self) -> str:
        return (
            "SamplingMetadata("
            f"seq_groups={self.seq_groups}, "
            f"seq_data={self.seq_data}, "
            f"prompt_lens={self.prompt_lens}, "
            f"selected_token_indices={self.selected_token_indices}, "
            f"categorized_sample_indices={self.categorized_sample_indices}, "
            f"persistent_metadata={self.persistent_metadata}, "
            f"output_metadata={self.output_metadata})")

@dataclass
class SamplingTensors:
    """Tensors for sampling."""
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    top_as: torch.Tensor
    min_ps: torch.Tensor
    presence_penalties: torch.Tensor
    frequency_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    tfss: torch.Tensor
    eta_cutoffs: torch.Tensor
    epsilon_cutoffs: torch.Tensor
    typical_ps: torch.Tensor
    prompt_tokens: torch.Tensor
    output_tokens: torch.Tensor

    @classmethod
    def from_sampling_metadata(
        cls, sampling_metadata: "SamplingMetadata", vocab_size: int,
        device: torch.device, dtype: torch.dtype
        ) -> Tuple["SamplingTensors", bool, bool, bool]:
        prompt_tokens: List[List[int]] = []
        output_tokens: List[List[int]] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        top_as: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        tfss: List[float] = []
        eta_cutoffs: List[float] = []
        epsilon_cutoffs: List[float] = []
        typical_ps: List[float] = []
        do_penalties = False
        do_alphabet_soup = False
        do_cutoffs = False
        for i, seq_group in enumerate(sampling_metadata.seq_groups):
            seq_ids, sampling_params = seq_group
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            # k should not be greater than the vocab size
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            top_a = sampling_params.top_a
            min_p = sampling_params.min_p
            tfs = sampling_params.tfs
            eta_cutoff = sampling_params.eta_cutoff
            epsilon_cutoff = sampling_params.epsilon_cutoff
            typical_p = sampling_params.typical_p
            if temperature < _SAMPLING_EPS:
                # NOTE: Zero temp means deterministic sampling
                # i.e. greedy sampling or beam search
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            if not do_alphabet_soup and (top_p < 1.0 - _SAMPLING_EPS
                                         or top_k != vocab_size
                                         or top_a > 0.0
                                         or min_p > _SAMPLING_EPS):
                do_alphabet_soup = True
            if not do_penalties and (abs(p) >= _SAMPLING_EPS
                                     or abs(f) >= _SAMPLING_EPS
                                     or abs(r - 1.0) >= _SAMPLING_EPS):
                do_penalties = True
            if not do_cutoffs and (eta_cutoff > _SAMPLING_EPS
                                   or epsilon_cutoff > _SAMPLING_EPS
                                   or typical_p < 1.0 - _SAMPLING_EPS
                                   or tfs < 1.0 - _SAMPLING_EPS):
                do_cutoffs = True
            if (i < sampling_metadata.num_prompts
                    and sampling_params.prompt_logprobs is not None):
                # For tokens in the prompt that we only need to get their
                # logprobs
                prompt_len = sampling_metadata.prompt_lens[i]
                temperatures += [temperature] * (prompt_len - 1)
                top_ps += [top_p] * (prompt_len - 1)
                top_ks += [top_k] * (prompt_len - 1)
                top_as += [top_a] * (prompt_len - 1)
                min_ps += [min_p] * (prompt_len - 1)
                presence_penalties += [0] * (prompt_len - 1)
                frequency_penalties += [0] * (prompt_len - 1)
                repetition_penalties += [1] * (prompt_len - 1)
                tfss += [1] * (prompt_len - 1)
                eta_cutoffs += [0] * (prompt_len - 1)
                epsilon_cutoffs += [0] * (prompt_len - 1)
                typical_ps += [1] * (prompt_len - 1)
                prompt_tokens.extend([] for _ in range(prompt_len - 1))
                output_tokens.extend([] for _ in range(prompt_len - 1))
            for seq_id in seq_ids:
                seq_data = sampling_metadata.seq_data[seq_id]
                prompt_tokens.append(seq_data.prompt_tokens)
                output_tokens.append(seq_data.output_tokens)
            temperatures += [temperature] * len(seq_ids)
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
            top_as += [top_a] * len(seq_ids)
            min_ps += [min_p] * len(seq_ids)
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
            repetition_penalties += [r] * len(seq_ids)
            tfss += [tfs] * len(seq_ids)
            eta_cutoffs += [eta_cutoff] * len(seq_ids)
            epsilon_cutoffs += [epsilon_cutoff] * len(seq_ids)
            typical_ps += [typical_p] * len(seq_ids)

        sampling_tensors = SamplingTensors.from_lists(
            temperatures, top_ps, top_ks, top_as, min_ps, presence_penalties,
            frequency_penalties, repetition_penalties, tfss, eta_cutoffs,
            epsilon_cutoffs, typical_ps, prompt_tokens, output_tokens,
            vocab_size, device, dtype)
        return (sampling_tensors, do_penalties, do_alphabet_soup, do_cutoffs)

    @classmethod
    def from_lists(cls, temperatures: List[float], top_ps: List[float],
                   top_ks: List[int], top_as: List[float], min_ps: List[float],
                   presence_penalties: List[float],
                   frequency_penalties: List[float],
                   repetition_penalties: List[float], tfss: List[float],
                   eta_cutoffs: List[float], epsilon_cutoffs: List[float],
                   typical_ps: List[float], prompt_tokens: List[List[int]],
                   output_tokens: List[List[int]], vocab_size: int,
                   device: torch.device, dtype: torch.dtype
                   ) -> "SamplingTensors":
        # Note that the performance will be very bad without
        # pinned memory.
        pin_memory = not in_wsl()
        prompt_max_len = max(len(tokens) for tokens in prompt_tokens)
        prompt_padded_tokens = [
            tokens + [vocab_size] * (prompt_max_len - len(tokens))
            for tokens in prompt_tokens
        ]
        output_max_len = max(len(tokens) for tokens in output_tokens)
        output_padded_tokens = [
            tokens + [vocab_size] * (output_max_len - len(tokens))
            for tokens in output_tokens
        ]

        temperatures_t = torch.tensor(
            temperatures,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        top_ps_t = torch.tensor(
            top_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        top_ks_t = torch.tensor(
            top_ks,
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory
        )
        top_as_t = torch.tensor(
            top_as,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        min_ps_t = torch.tensor(
            min_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        presence_penalties_t = torch.tensor(
            presence_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        frequency_penalties_t = torch.tensor(
            frequency_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        repetition_penalties_t = torch.tensor(
            repetition_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        tfss_t = torch.tensor(
            tfss,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        eta_cutoffs_t = torch.tensor(
            eta_cutoffs,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        epsilon_cutoffs_t = torch.tensor(
            epsilon_cutoffs,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        typical_ps_t = torch.tensor(
            typical_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory
        )
        prompt_tensor = torch.tensor(
            prompt_padded_tokens,
            device=device,
            dtype=torch.long,
            pin_memory=pin_memory
        )
        output_tensor = torch.tensor(
            output_padded_tokens,
            device=device,
            dtype=torch.long,
            pin_memory=pin_memory
        )
        # Because the memory is pinned, we can do non-blocking
        # transfer to device.
        return cls(
            temperatures=temperatures_t.to(device=device, non_blocking=True),
            top_ps=top_ps_t.to(device=device, non_blocking=True),
            top_ks=top_ks_t.to(device=device, non_blocking=True),
            top_as=top_as_t.to(device=device, non_blocking=True),
            min_ps=min_ps_t.to(device=device, non_blocking=True),
            presence_penalties=presence_penalties_t.to(
                device=device, non_blocking=True),
            frequency_penalties=frequency_penalties_t.to(
                device=device, non_blocking=True),
            repetition_penalties=repetition_penalties_t.to(
                device=device, non_blocking=True),
            tfss=tfss_t.to(device=device, non_blocking=True),
            eta_cutoffs=eta_cutoffs_t.to(device=device, non_blocking=True),
            epsilon_cutoffs=epsilon_cutoffs_t.to(
                device=device, non_blocking=True),
            typical_ps=typical_ps_t.to(device=device, non_blocking=True),
            prompt_tokens=prompt_tensor.to(device=device, non_blocking=True),
            output_tokens=output_tensor.to(device=device, non_blocking=True),
        )
