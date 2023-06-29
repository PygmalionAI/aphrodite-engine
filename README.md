<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](./assets/aphrodite.png)

Aphrodite is the official backend engine for PygmalionAI. It is designed to serve as the inference endpoint for the PygmalionAI website, and to allow serving the [Pygmalion](https://huggingface.co/PygmalionAI) models to a large number of users with blazing fast speeds (thanks to FasterTransformer). 

Aphrodite builds upon and integrates the exceptional work from various projects, including:


- [vLLM](https://github.com/vllm-project/vllm) (CacheFlow)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FastChat](https://github.com/lm-sys/FastChat)
- [SkyPilot](https://github.com/skypilot-org/skypilot)
- [OpenAI Python Library](https://github.com/openai/openai-python)

<h2>Please note that Aphrodite is currently in active development and not yet fully functional.</h2>

## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](./aphrodite/modeling/layers/attention.py)
- Optimized CUDA kernels for improved inference
- Distributed inference
- Multiple decoding algorithms (e.g. parallel sampling, beam search)


## Requirements

- Operating System: Linux
- Python: at least 3.8

## Supported GPUs

Basically, anything with a compute capability of 7.0 or higher. Here's a full list of supported consumer GPUs:

| GPU     | CC  | GPU       | CC  | GPU     | CC  |
| ------- | --- | --------- | --- | ------- | --- |
| 2060    | 7.5 | 2070      | 7.5 | 2080    | 7.5 |
| 2080 Ti | 7.5 | Titan RTX | 7.5 | 1650 Ti | 7.5 |
| 3060    | 8.6 | 3060 Ti   | 8.6 | 3070    | 8.6 |
| 3070 Ti | 8.6 | 3080      | 8.6 | 3080 Ti | 8.6 |
| 3090    | 8.6 | 3090 Ti   | 8.6 | 4070 Ti | 8.9 |
| 4080    | 8.9 | 4090      | 8.9 |         |     |

> `*` `CC`: Compute Capability

If your GPU isn't listed here, you won't be able to run Aphrodite.

## Usage
***Currently not working, but this is how you'd run it once it's fixed.**
- Clone the repository:
  ```sh
  git clone https://github.com/PygmalionAI/aphrodite-engine && cd aphrodite-engine
  ```
- Install the package:
  ```
  pip install -r requirements.txt
  pip install -e .
  ```
- Example usage:
  ```py
  from aphrodite import LLM, SamplingParams

  prompts = [
    "What is a man? A",
    "The sun is a wondrous body, like a magnificent",
    "All flesh is grass and all the comeliness thereof",
  ]
  sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

  llm = LLM(model="PygmalionAI/pygmalion-350m")
  outputs = llm.generate(prompts, sampling_params)
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
  ```


## Contributing
We accept PRs! There will likely be a few typos or other errors we've failed to catch, so please let us know either via an issue or make a Pull Request.
