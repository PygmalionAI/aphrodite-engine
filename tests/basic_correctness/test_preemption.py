"""Compare the short outputs of HF and Aphrodite when using greedy sampling.

APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 has to be set before running this test.

Run `APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT=1
pytest tests/basic_correctness/test_preemption.py`.
"""
import pytest
from prometheus_client import REGISTRY

from aphrodite import SamplingParams
from aphrodite.core.scheduler import (ARTIFICIAL_PREEMPTION_MAX_CNT,
                                      ENABLE_ARTIFICIAL_PREEMPT)

from ..models.utils import check_outputs_equal

MODELS = [
    "facebook/opt-125m",
]

assert ENABLE_ARTIFICIAL_PREEMPT is True, (
    "Use an env var APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT=1. "
    "`APHRODITE_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 pytest "
    "tests/basic_correctness/test_preemption.py`")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("chunked_prefill_token_size", [16])
def test_chunked_prefill_recompute(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
) -> None:
    """Ensure that chunked prefill works with preemption."""
    max_num_seqs = min(chunked_prefill_token_size, 256)
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_batched_tokens = chunked_prefill_token_size

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with aphrodite_runner(
            model,
            dtype=dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_seqs=max_num_seqs,
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy(example_prompts, max_tokens)
        assert (aphrodite_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        aphrodite_output_ids, aphrodite_output_str = aphrodite_outputs[i]
        assert hf_output_str == aphrodite_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nAphrodite: {aphrodite_output_str!r}")
        assert hf_output_ids == aphrodite_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nAphrodite: {aphrodite_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_preemption(
    caplog_aphrodite,
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """By default, recompute preemption is enabled"""

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with aphrodite_runner(
            model,
            dtype=dtype,
            disable_log_stats=False,
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy(example_prompts, max_tokens)
        assert (aphrodite_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)
        total_preemption = (
            aphrodite_model.model.llm_engine.scheduler[0].num_cumulative_preemption)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )

    assert ("is preempted by PreemptionMode.RECOMPUTE mode because there "
            "is not enough KV cache space." in caplog_aphrodite.text)
    # Ensure the count bucket of request-level histogram metrics matches
    # the number of requests as a simple sanity check to ensure metrics are
    # generated
    preemption_metrics = None
    for m in REGISTRY.collect():
        if m.name == "aphrodite:num_preemptions":
            preemption_metrics = m
    assert preemption_metrics is not None
    total_recorded_preemption = 0
    for sample in preemption_metrics.samples:
        total_recorded_preemption += sample.value
    assert total_preemption == total_recorded_preemption


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("beam_width", [4])
def test_swap(
    caplog_aphrodite,
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    """Use beam search enables swapping."""
    example_prompts = example_prompts[:1]
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width,
                                                   max_tokens)

    with aphrodite_runner(
            model,
            dtype=dtype,
            swap_space=10,
            disable_log_stats=False,
    ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_beam_search(example_prompts,
                                                       beam_width, max_tokens)
        assert (aphrodite_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)
        total_preemption = (
            aphrodite_model.model.llm_engine.scheduler[0].num_cumulative_preemption)

    for i in range(len(example_prompts)):
        hf_output_ids, _ = hf_outputs[i]
        aphrodite_output_ids, _ = aphrodite_outputs[i]
        assert len(hf_output_ids) == len(aphrodite_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == aphrodite_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"Aphrodite: {aphrodite_output_ids}")

    assert ("is preempted by PreemptionMode.SWAP mode because there "
            "is not enough KV cache space." in caplog_aphrodite.text)
    # Ensure the count bucket of request-level histogram metrics matches
    # the number of requests as a simple sanity check to ensure metrics are
    # generated
    preemption_metrics = None
    for m in REGISTRY.collect():
        if m.name == "aphrodite:num_preemptions":
            preemption_metrics = m
    assert preemption_metrics is not None
    total_recorded_preemption = 0
    for sample in preemption_metrics.samples:
        total_recorded_preemption += sample.value
    assert total_preemption == total_recorded_preemption


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("beam_width", [4])
def test_swap_infeasible(
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    """Verify infeasible swap request will be ignored."""
    BLOCK_SIZE = 16
    prefill_blocks = 2
    decode_blocks = max_tokens // BLOCK_SIZE
    example_prompts = example_prompts[:1]

    with aphrodite_runner(
            model,
            dtype=dtype,
            swap_space=10,
            block_size=BLOCK_SIZE,
            # Since beam search have more than 1 sequence, prefill +
            # decode blocks are not enough to finish.
            num_gpu_blocks_override=prefill_blocks + decode_blocks,
            max_model_len=(prefill_blocks + decode_blocks) * BLOCK_SIZE,
    ) as aphrodite_model:
        sampling_params = SamplingParams(n=beam_width,
                                         use_beam_search=True,
                                         temperature=0.0,
                                         max_tokens=max_tokens,
                                         ignore_eos=True)
        req_outputs = aphrodite_model.model.generate(
            example_prompts,
            sampling_params=sampling_params,
        )
        assert (aphrodite_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)

    # Verify the request is ignored and not hang.
    assert req_outputs[0].outputs[0].finish_reason == "length"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_preemption_infeasible(
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """Verify infeasible preemption request will be ignored."""
    BLOCK_SIZE = 16
    prefill_blocks = 2
    decode_blocks = max_tokens // BLOCK_SIZE
    with aphrodite_runner(
            model,
            dtype=dtype,
            block_size=BLOCK_SIZE,
            # Not enough gpu blocks to complete a single sequence.
            # preemption should happen, and the sequence should be
            # ignored instead of hanging forever.
            num_gpu_blocks_override=prefill_blocks + decode_blocks // 2,
            max_model_len=((prefill_blocks + decode_blocks // 2) * BLOCK_SIZE),
    ) as aphrodite_model:
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         ignore_eos=True)
        req_outputs = aphrodite_model.model.generate(
            example_prompts,
            sampling_params=sampling_params,
        )

        assert (aphrodite_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)

    # Verify the request is ignored and not hang.
    for req_output in req_outputs:
        outputs = req_output.outputs
        assert len(outputs) == 1
        assert outputs[0].finish_reason == "length"
