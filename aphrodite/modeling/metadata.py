from typing import Dict, List, Tuple, Optional
import torch
from xformers.ops import AttentionBias

from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import SequenceData


class PersistentMetadata:

    def __init__(self):
        self._metadata: dict[int, dict] = {}

    def get(self, seq_id: int) -> dict:
        return self._metadata.get(seq_id, {})


class OutputMetadata(PersistentMetadata):

    def add(self, seq_id: int, key, val) -> None:
        if seq_id not in self._metadata:
            self._metadata[seq_id] = {}
        self._metadata[seq_id][key] = val


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        block_tables: torch.Tensor,
        sliding_window: Optional[int] = None,
        persistent_data: Optional[PersistentMetadata] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables
        self.persistent_data = persistent_data or PersistentMetadata()

        self.max_prompt_len = max(prompt_lens) if prompt_lens else 0
        self.to_cache = None
        if sliding_window is not None:
            # We need to keep the positions of sliding windows within
            # the key/value tables, this is helpful to know which
            # elements we need to cache and where.
            to_cache, start_idx = [], 0
            for prompt_len in self.prompt_lens:
                to_cache.extend(
                    range(
                        start_idx + max(0, prompt_len - sliding_window),
                        start_idx + prompt_len,
                    ))
                start_idx += self.max_prompt_len
            to_cache.extend(range(start_idx, slot_mapping.shape[0]))
            self.to_cache = torch.tensor(to_cache,
                                         dtype=torch.int32,
                                         device=self.slot_mapping.device)

        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = self.num_prompts * self.max_prompt_len
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = slot_mapping.shape[0]
        if block_tables.numel() > 0:
            self.max_num_blocks_per_seq = block_tables.shape[1]
        else:
            self.max_num_blocks_per_seq = 0
        assert block_tables.shape[0] == self.num_generation_tokens

        # Set during the execution of the first attention op.
        self.attn_bias: Optional[AttentionBias] = None

    def __repr__(self) -> str:
        # Print only useful metadata.
        return (f'InputMetadata('
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len}, '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'block_tables={self.block_tables}, '
                f'slot_mapping={self.slot_mapping}, '
                f'persistent_data={self.persistent_data})')
