import enum
import time
from typing import Dict, List, Optional, Tuple

from aphrodite.config import CacheConfig, SchedulerConfig
from aphrodite.processing.block_manager import BlockSpaceManager
from aphrodite.processing.policy import PolicyFactory
from aphrodite.logger import init_logger
from aphrodite.sequence import (Sequence, SequenceData, SequenceGroup,
                                SequenceGroupMetadata, SequenceOutputs, SequenceStatus)

logger = init_logger(__name__)

__LOGGING_INTERVAL_SEC = 5

class PreemptionMode(enum.Enum):
    """Preemtion modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory and
    swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and recompute
    them when the sequences are resumed, treating the sequences as new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()

class SchedulerOutputs:

    def __init__(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        assert not (blocks_to_swap_in and blocks_to_swap_out)

    def is_empty(self) -> bool:
        return (not self.blocks_to_swap_in and not self.blocks_to_swap_out and not self.blocks_to_copy)
    

class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        log_stats: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config - cache_config
        self.log_stats = log_stats

        self.policy = PolicyFactory.get_policy(policy_name='fcfs')
        self.block_manager = BlockingSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        self.waiting: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

        self.last_logging_time: float = 0.0
        self.num_input_tokens: List[Tuple[float, int]] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group in state_queue:
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)
    
    def _schedule(self) -> Tuple[SchedulerOutputs, List[str]]:
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        now time.time()

        """
        NOTE: We prioritize the sequence groups in the RUNNING state in order to
        minimize the preemption overheads.
        Preemption happens only when there's no available slot to keep all the
        sequence groups in the RUNNING state.
        In this case the policy is responsible for deciding which sequence groups to
        preempt.
        """
        self.running = self.policy.sort_by_priority(now, self.running)

        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                self.append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            if seq_group in preempted:
                break
            if not self.block_manager.can_swap_in(seq_group):
                break

            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = len(self.running)
            if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )
        
        prompt_group_ids: List[str] = []
        """
        NOTE: The sequence groups in the SWAPPED state are strictly prioritized
        over the sequence groups in the WAITING state.
        This is because we want to bound the amount of CPU memory taken by the
        swapped sequence groups.
        """
        if not self.swapped:
            """
            NOTE(optimization): We don't sort the waiting queue since the preempted sequence
            groups are added to the front and the new sequence groups are added to the back.
            """
            while self.waiting:
                seq_group = self.waiting[0]
                if seq_group in preempted:
                    break
                if not self.block_manager.can_allocate(seq_group):
                    break
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if (num_batched_tokens + num_prompt_tokens > self.scheduler_config.max_num_batched_tokens):
                    break

                num_new_seqs = seq_group.num_seqs(status=SequenceStatus.WAITING)
                num_curr_seqs = len(self.running)
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                prompt_group_ids.append(seq_group.request_id)
        
        scheduler_outputs = SchedulerOutputs(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        if not self.log_stats:
            return scheduler_outputs, prompt_group_ids
            """WORK IN PROGRESS"""
