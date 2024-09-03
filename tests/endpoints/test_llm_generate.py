import pytest

from aphrodite import LLM, SamplingParams


def test_multiple_sampling_params():
    llm = LLM(model='gpt2', max_num_batched_tokens=1024)
    prompts = [
        "Once upon a time",
        "In a galaxy far far away",
        "The quick brown fox jumps over the lazy dog",
    ]
    sampling_params = [
        SamplingParams(temperature=0.7, min_p=0.06),
        SamplingParams(temperature=0.8, min_p=0.07),
        SamplingParams(temperature=0.9, min_p=0.08),
    ]

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(prompts) == len(outputs)

    with pytest.raises(ValueError):
        outputs = llm.generate(prompts, sampling_params=sampling_params[:2])

        single_sampling_params = SamplingParams(temperature=0.7, min_p=0.06)
        outputs = llm.generate(prompts, sampling_params=single_sampling_params)
        assert len(prompts) == len(outputs)

        outputs = llm.generate(prompts, sampling_params=None)
        assert len(prompts) == len(outputs)
