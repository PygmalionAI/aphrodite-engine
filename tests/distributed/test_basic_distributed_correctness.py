"""Compare the outputs of HF and distributed Aphrodite when using greedy sampling.
Aphrodite will allocate all the available memory, so we need to run the tests one
by one. The solution is to pass arguments (model name) by environment
variables.
Run:
```sh
TEST_DIST_MODEL=alpindale/gemma-2b pytest \
    test_basic_distributed_correctness.py
TEST_DIST_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
    test_basic_distributed_correctness.py
```
"""
import os

import pytest
import torch

MODELS = [
    os.environ["TEST_DIST_MODEL"],
]


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    aphrodite_model = aphrodite_runner(
        model,
        dtype=dtype,
        tensor_parallel_size=2,
    )
    aphrodite_outputs = aphrodite_model.generate_greedy(example_prompts,
                                                        max_tokens)
    del aphrodite_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        aphrodite_output_ids, aphrodite_output_str = aphrodite_outputs[i]
        assert hf_output_str == aphrodite_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nAphrodite: "
            f"{aphrodite_output_str!r}")
        assert hf_output_ids == aphrodite_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nAphrodite: {aphrodite_output_ids}")