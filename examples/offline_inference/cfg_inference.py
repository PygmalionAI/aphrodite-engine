from typing import List

from aphrodite import LLM, SamplingParams
from aphrodite.inputs import PromptInputs

llm = LLM(
    model="NousResearch/Meta-Llama-3.1-8B-Instruct",
    use_v2_block_manager=True,
    cfg_model="NousResearch/Meta-Llama-3.1-8B-Instruct",
    max_model_len=8192,
)

prompt_pairs = [
    {
        "prompt": "Hello, my name is",
        "negative_prompt": "I am uncertain and confused about who I am"
    },
    {
        "prompt": "The president of the United States is",
        "negative_prompt": "I don't know anything about US politics or leadership"  # noqa: E501
    },
]

tokenizer = llm.get_tokenizer()

inputs: List[PromptInputs] = [
    {
        "prompt_token_ids": tokenizer.encode(text=pair["prompt"]),
        "negative_prompt_token_ids": tokenizer.encode(text=pair["negative_prompt"])  # noqa: E501
    }
    for pair in prompt_pairs
]

sampling_params = SamplingParams(guidance_scale=5.0, max_tokens=128)
outputs = llm.generate(inputs, sampling_params)

for i, output in enumerate(outputs):
    prompt_pair = prompt_pairs[i]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt_pair['prompt']!r}")
    print(f"Negative Prompt: {prompt_pair['negative_prompt']!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)