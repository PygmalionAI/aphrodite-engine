import pytest
import torch

from aphrodite import SamplingParams

MODELS = ["EleutherAI/pythia-70m-deduped"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_get_prompt_logprobs(
    hf_runner,
    aphrodite_runner,
    model,
    dtype,
    example_prompts,
):
    max_tokens = 5
    hf_model = hf_runner(model, dtype=dtype)
    hf_logprobs = hf_model.generate_greedy_logprobs(
        example_prompts,
        max_tokens=max_tokens,
    )
    del hf_model

    aphrodite_model = aphrodite_runner(model, dtype=dtype)
    aphrodite_sampling_params = SamplingParams(max_tokens=max_tokens,
                                               logprobs=5,
                                               prompt_logprobs=5,
                                               temperature=0.0)
    aphrodite_results = aphrodite_model.model.generate(
        example_prompts, sampling_params=aphrodite_sampling_params)

    # Test whether logprobs are included in the results.
    for result in aphrodite_results:
        assert result.prompt_logprobs is not None
        assert result.outputs[0].logprobs is not None

    # Test whether prompt logprobs are consistent with HF
    for aphrodite_result, hf_logprob in zip(aphrodite_results, hf_logprobs):
        # Check prompt logprobs
        aphrodite_prompt_logprobs = aphrodite_result.prompt_logprobs[1:]
        for i, aphrodite_prompt_logprob_dict in enumerate(
                aphrodite_prompt_logprobs):
            for token_id, logprob in aphrodite_prompt_logprob_dict.items():
                torch.testing.assert_close(logprob,
                                           hf_logprob[0][i][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
        aphrodite_sample_logprobs = aphrodite_result.outputs[0].logprobs
        for i, aphrodite_sample_logprob_dict in enumerate(
                aphrodite_sample_logprobs):
            for token_id, logprob in aphrodite_sample_logprob_dict.items():
                torch.testing.assert_close(logprob,
                                           hf_logprob[i][-1][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
