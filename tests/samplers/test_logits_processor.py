import pytest
import torch

from aphrodite import SamplingParams

MODELS = ["facebook/opt-125m"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_logits_processor_force_generate(
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with aphrodite_runner(model, dtype=dtype) as aphrodite_model:
        tokenizer = aphrodite_model.model.get_tokenizer()
        repeat_times = 2
        enforced_answers = " aphrodite"
        aphrodite_token_ids = tokenizer.encode(enforced_answers,
                                          add_special_tokens=False)
        max_tokens = len(aphrodite_token_ids) * repeat_times

        def pick_aphrodite(token_ids, logits):
            token_id = aphrodite_token_ids[len(token_ids) %
                                           len(aphrodite_token_ids)]
            logits[token_id] = torch.finfo(logits.dtype).max
            return logits

        params_with_logprobs = SamplingParams(
            logits_processors=[pick_aphrodite],
            prompt_logprobs=3,
            max_tokens=max_tokens,
        )

        # test logits_processors when prompt_logprobs is not None
        aphrodite_model.model._add_request(
            example_prompts[0],
            params=params_with_logprobs,
        )

        # test prompt_logprobs is not None
        aphrodite_model.model._add_request(
            example_prompts[1],
            params=SamplingParams(
                prompt_logprobs=3,
                max_tokens=max_tokens,
            ),
        )

        # test grouped requests
        aphrodite_model.model._add_request(
            example_prompts[2],
            params=SamplingParams(max_tokens=max_tokens),
        )

        outputs = aphrodite_model.model._run_engine(use_tqdm=False)

        assert outputs[0].outputs[0].text == enforced_answers * repeat_times
