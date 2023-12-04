import pytest
import random
from typing import Tuple
from unittest.mock import patch

import torch

from aphrodite.modeling.layers.sampler import Sampler
from aphrodite.modeling.utils import set_random_seed
from aphrodite.common.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from aphrodite.task_handler.worker import Worker


class MockLogitsSampler(Sampler):

    def __init__(self, vocab_size: int, fake_logits: torch.Tensor):
        super().__init__(vocab_size=vocab_size)
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        with patch("aphrodite.modeling.layers.sampler._prune_hidden_states",
                   lambda x, y: x):
            with patch("aphrodite.modeling.layers.sampler._get_logits",
                       lambda *args, **kwargs: self.fake_logits):
                return super().forward(*args, **kwargs)


def _prepare_test(
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler, Worker]:
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024),
                              device="cuda",
                              dtype=torch.float16)
    fake_logits = torch.full((batch_size, vocab_size),
                             1e-2,
                             device=input_tensor.device,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(32000, fake_logits)
    worker = Worker(None, None, None)
    worker.block_size = 16
    return input_tensor, fake_logits, sampler, worker


RANDOM_SEEDS = list(range(128))


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_greedy(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    seq_group_metadata_list = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(request_id=f"test_{i}",
                                  is_prompt=True,
                                  seq_data={0: SequenceData([1, 2, 3])},
                                  sampling_params=SamplingParams(
                                      temperature=0, ),
                                  block_tables={0: [1]},
                                  persistent_data={}))

    # pylint: disable=protected-access
    _, _, input_metadata = worker._prepare_inputs(seq_group_metadata_list)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             input_metadata=input_metadata)
    expected = torch.argmax(fake_logits, dim=-1)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output:
            assert nth_output.output_token == expected[i].item()


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_random(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    seq_group_metadata_list = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(request_id=f"test_{i}",
                                  is_prompt=True,
                                  seq_data={0: SequenceData([1, 2, 3])},
                                  sampling_params=SamplingParams(
                                      temperature=1.0,
                                      n=random.randint(1, 10),
                                  ),
                                  block_tables={0: [1]},
                                  persistent_data={}))
    # pylint: disable=protected-access
    _, _, input_metadata = worker._prepare_inputs(seq_group_metadata_list)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             input_metadata=input_metadata)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output:
            assert nth_output.output_token == i


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_beam(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    # pylint: disable=unused-variable
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    seq_group_metadata_list = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(request_id=f"test_{i}",
                                  is_prompt=True,
                                  seq_data={0: SequenceData([1, 2, 3])},
                                  sampling_params=SamplingParams(
                                      temperature=0,
                                      best_of=2,
                                      use_beam_search=True,
                                  ),
                                  block_tables={0: [1]},
                                  persistent_data={}))
    # pylint: disable=protected-access
    _, _, input_metadata = worker._prepare_inputs(seq_group_metadata_list)
    sampler(embedding=None,
            hidden_states=input_tensor,
            input_metadata=input_metadata)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_mixed(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    seq_group_metadata_list = []
    expected_tokens = []
    for i in range(batch_size):
        n = 1
        sampling_type = random.randint(0, 2)
        if sampling_type == 0:
            sampling_params = SamplingParams(temperature=0)
        elif sampling_type == 1:
            n = random.randint(1, 10)
            sampling_params = SamplingParams(
                temperature=random.random() + 0.1,
                top_p=min(random.random() + 0.1, 1),
                top_k=random.randint(0, 10) or -1,
                top_a=min(random.random() + 0.1, 2),
                tfs=min(random.random() + 0.1, 1),
                eta_cutoff=random.randint(0, 10) or 0,
                epsilon_cutoff=random.randint(0, 10) or 0,
                typical_p=min(random.random() + 0.1, 1),
                presence_penalty=random.randint(0, 1),
                frequency_penalty=random.randint(0, 1),
                repetition_penalty=min(random.random() + 0.1, 1),
                n=n,
            )
        else:
            sampling_params = SamplingParams(temperature=0,
                                             use_beam_search=True,
                                             best_of=2)

        for idx in range(n):
            fake_logits[i, i + idx] = 1e2
            expected_tokens.append(i + idx)
        seq_group_metadata_list.append(
            SequenceGroupMetadata(request_id=f"test_{i}",
                                  is_prompt=True,
                                  seq_data={0: SequenceData([1, 2, 3])},
                                  sampling_params=sampling_params,
                                  block_tables={0: [1]},
                                  persistent_data={}))

    # pylint: disable=protected-access
    _, _, input_metadata = worker._prepare_inputs(seq_group_metadata_list)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             input_metadata=input_metadata)
    for i, sequence_output in enumerate(sampler_output):
        if seq_group_metadata_list[i].sampling_params.use_beam_search:
            continue
        for nth_output in sequence_output:
            assert nth_output.output_token in expected_tokens
