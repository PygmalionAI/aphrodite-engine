"""Compare the outputs of HF and distributed Aphrodite when using greedy sampling.

Run:
```sh
cd $APHRODITE_PATH/tests

pytest distributed/test_basic_distributed_correctness.py
```
"""
import os

import pytest

from aphrodite.common.utils import cuda_device_count_stateless

from ..models.utils import check_outputs_equal
from ..utils import fork_new_process_for_each_test

TARGET_TEST_SUITE = os.environ.get("TARGET_TEST_SUITE", "L4")


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "model, distributed_executor_backend, attention_backend, test_suite", [
        ("facebook/opt-125m", "ray", "", "L4"),
        ("facebook/opt-125m", "mp", "", "L4"),
        ("meta-llama/Llama-2-7b-hf", "ray", "", "L4"),
        ("meta-llama/Llama-2-7b-hf", "mp", "", "L4"),
        ("facebook/opt-125m", "ray", "", "A100"),
        ("facebook/opt-125m", "mp", "", "A100"),
        ("facebook/opt-125m", "mp", "FLASHINFER", "A100"),
        ("meta-llama/Meta-Llama-3-8B", "ray", "FLASHINFER", "A100"),
    ])
@fork_new_process_for_each_test
def test_models(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
    test_suite: str,
) -> None:

    if test_suite != TARGET_TEST_SUITE:
        pytest.skip(f"Skip test for {test_suite}")

    if model == "meta-llama/Llama-2-7b-hf" and distributed_executor_backend == "ray" and attention_backend == "" and test_suite == "L4":  # noqa
        # test ray adag
        os.environ['APHRODITE_USE_RAY_SPMD_WORKER'] = "1"
        os.environ['APHRODITE_USE_RAY_COMPILED_DAG'] = "1"

    if attention_backend:
        os.environ["APHRODITE_ATTENTION_BACKEND"] = attention_backend

    dtype = "half"
    max_tokens = 5

    # NOTE: take care of the order. run Aphrodite first, and then run HF.
    # Aphrodite needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with aphrodite_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=2,
                     distributed_executor_backend=distributed_executor_backend
                     ) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy(example_prompts, max_tokens)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )
