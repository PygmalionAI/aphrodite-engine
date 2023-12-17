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
        do_penalties: False
        do_alphabet_soup: False
        do_cutoffs: False
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
            if not do_alphabet_soup and ()


