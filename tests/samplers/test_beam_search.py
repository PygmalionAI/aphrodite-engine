"""Compare the outputs of HF and Aphrodite Engine when using beam search.

Run `pytest tests/samplers/test_beam_search.py --forked`.
"""
import pytest

MAX_TOKENS = [128]
BEAM_WIDTHS = [4]
MODELS = ["EleutherAI/pythia-70m-deduped"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_single_input(
    hf_runner,
    aphrodite_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width,
                                               max_tokens)
    del hf_model

    aphrodite_model = aphrodite_runner(model, dtype=dtype)
    aphrodite_outputs = aphrodite_model.generate_beam_search(
        example_prompts, beam_width, max_tokens)
    del aphrodite_model

    for i in range(len(example_prompts)):
        hf_output_ids, _ = hf_outputs[i]
        aphrodite_output_ids, _ = aphrodite_outputs[i]
        assert len(hf_output_ids) == len(aphrodite_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == aphrodite_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"Aphrodite Engine: {aphrodite_output_ids}")
