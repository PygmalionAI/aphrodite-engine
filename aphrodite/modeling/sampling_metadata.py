from typing import Dict, List, Tuple, Optional

import torch

from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import SequenceData


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
