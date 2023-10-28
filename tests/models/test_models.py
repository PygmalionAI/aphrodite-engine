import pytest

MODELS = [
    "EleutherAI/gpt-j-6b",
    "EleutherAI/gpt-neox-20b",
    "meta-llama/llama-2-7b-hf",
    "/mistralai/Mistral-7B-v0.1",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
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

    aphrodite_model = aphrodite_runner(model, dtype=dtype)
    aphrodite_outputs = aphrodite_model.generate_greedy(
        example_prompts, max_tokens)
    del aphrodite_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        aphrodite_output_ids, aphrodite_output_str = aphrodite_outputs[i]
        assert hf_output_str == aphrodite_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nAphrodite: {aphrodite_output_str!r}"
        )
        assert hf_output_ids == aphrodite_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nAphrodite: {aphrodite_output_ids}"
        )
