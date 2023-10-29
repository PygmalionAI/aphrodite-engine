import random

import pytest
import torch

from aphrodite import cache_ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]
NUM_LAYERS = [5]
NUM_HEADS = [8]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [8, 16, 32]
NUM_BLOCKS = [1024]
NUM_MAPPINGS = [32, 256]
SEEDS = [0]

@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_copy_blocks(
    kv_cache_factory,
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remaining_blocks, 2 * num_mappings)
    block_mapping = {}
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping[src] = [dst1, dst2]
    
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, dtype, seed)
    cloned_key_caches = [key_caches.clone() for key_cache in key_caches]
    cloned_value_caches = [value_caches.clone() for value_cache in value_caches]

    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key_cache in cloned_key_caches:
                cloned_key_cache[dst] = cloned_key_cache[src]
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst] = cloned_value_cache[src]
    
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_key_cache in zip(value_caches,
                                             cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device='cuda')

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    _, key, value = qkv.unbind(dim=1)

    # create the KV caches
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # clone the KV caches
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # call the reshape_and_cache kernel
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping)
    
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode='floor')
    block_indicies = block_indicies.cpu().tolist()
    block_offset = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offset[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)


