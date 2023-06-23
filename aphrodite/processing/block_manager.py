"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set, Tuple, Any
from aphrodite.block import PhysicalTokenBlock
from aphrodite.sequence import Sequence, SequenceGroup, SequenceStatus
from aphrodite.utils import Device

class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(
                device=device, block_number=1, block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out Of Memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

BlockTable = List[PhysicalTokenBlock]

class BlockSpaceManager:
    """Manages the mapping between logical and physical blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(DEVICE.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(DEVICE.CPU, block_size, num_cpu_blocks)

        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        """
        NOTE: we assume that all sequences in the group share the same prompt.
        This might not be true for preempted sequences. Needs fixing.
        """
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, seq_group: SequenceGroup) -> None:
        seq = seq_group.get_seqs()[0]

        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        """
        Simple heuristic: If there's at least one free block
        for each sequence, we can append.
        """
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seq = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, req: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token"""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            return None
        else:
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number
    
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        src_block_table = self.block_size[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            block_table = self.block_size[seq.seq_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)


    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_free_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping
    
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate
                    maping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_table_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in block_table:
            if block.device == DEVICE.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table[block_table]
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()