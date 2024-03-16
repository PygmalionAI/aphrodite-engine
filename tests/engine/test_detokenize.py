import pytest

from transformers import AutoTokenizer

from aphrodite.transformers_utils.tokenizer import detokenize_incrementally

TRUTH = [
    "Tell me your favorite story.",
    "Transformers have revolutionized almost all natural language processing (NLP) tasks but suffer from memory and computational complexity that scales quadratically with sequence length. In contrast, recurrent neural networks (RNNs) exhibit linear scaling in memory and computational requirements but struggle to match the same performance as Transformers due to limitations in parallelization and scalability. We propose a novel model architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs. Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an RNN, which parallelizes computations during training and maintains constant computational and memory complexity during inference, leading to the first non-transformer architecture to be scaled to tens of billions of parameters. Our experiments reveal that RWKV performs on par with similarly sized Transformers, suggesting that future work can leverage this architecture to create more efficient models. This work presents a significant step towards reconciling the trade-offs between computational efficiency and model performance in sequence processing tasks."  # noqa: E501
    "トランスフォーマーは、ほぼすべての自然言語処理に革命をもたらしました",
]

TOKENIZERS = [
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m-deduped",
    "meta-llama/llama-2-7b-hf",
    "/mistralai/Mistral-7B-v0.1",
]


def _run_incremental_decode(tokenizer, all_input_ids,
                            skip_special_tokens: bool):
    decoded_text = ""
    offset = 0
    token_offset = 0
    prev_tokens = None
    for i in range(len(all_input_ids)):
        new_tokens, text, offset, token_offset = detokenize_incrementally(
            tokenizer,
            all_input_ids[:i + 1],
            prev_tokens,
            offset,
            token_offset,
            skip_special_tokens=skip_special_tokens)
        decoded_text += text
        if prev_tokens is None:
            prev_tokens = new_tokens
        else:
            prev_tokens += new_tokens
    return decoded_text


@pytest.mark.parametrize("truth", TRUTH)
@pytest.mark.parametrize("tokenizer_id", TOKENIZERS)
@pytest.mark.parametrize("skip_special_tokens", (True, False))
def test_decode_streaming(tokenizer_id, truth, skip_special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    all_input_ids = tokenizer(truth, add_special_tokens=False)["input_ids"]
    if skip_special_tokens:
        all_input_ids = (
            [tokenizer_id.bos_token_id] if tokenizer.bos_token_id is not None
            else []) + all_input_ids + [tokenizer.eos_token_id]  # type: ignore
    decoded_text = _run_incremental_decode(
        tokenizer, all_input_ids, skip_special_tokens=skip_special_tokens)

    assert decoded_text == truth
