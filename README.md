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
- Multiple decoding algorithsm (e.g. parallel sampling, beam search)


## Requirements

You will likely need a CUDA version of at least 11.0, and a Compute Capability of at least `7, 0`. CUDA 12.0 is unsupported, so please switch to 11.8!

**Linux-only**

## Contributing
We accept PRs! There will likely be a few typos or other errors we've failed to catch, so please let us know either via an issue or make a Pull Request.
