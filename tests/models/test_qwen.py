from typing import Type

import pytest

from ..conftest import AphroditeRunner, HfRunner
from .utils import check_logprobs_close

models = ["qwen/qwen-vl"]


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("model", models)
def test_text_only_qwen_model(
    hf_runner: Type[HfRunner],
    aphrodite_runner: Type[AphroditeRunner],
    example_prompts,
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    # This test checks language inputs only, since the visual component
    # for qwen-vl is still unsupported in Aphrodite. In the near-future, the
    # implementation and this test will be extended to consider
    # visual inputs as well.
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts,
            max_tokens,
            num_logprobs=num_logprobs,
        )

    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        aphrodite_outputs = aphrodite_model.generate_greedy_logprobs(
            example_prompts,
            max_tokens,
            num_logprobs=num_logprobs,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=aphrodite_outputs,
        name_0="hf",
        name_1="aphrodite",
    )
