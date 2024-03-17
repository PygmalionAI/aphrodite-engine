<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](https://raw.githubusercontent.com/PygmalionAI/aphrodite-engine/main/assets/aphrodite.png)

Aphrodite is the official backend engine for PygmalionAI. It is designed to serve as the inference endpoint for the PygmalionAI website, and to allow serving the [Pygmalion](https://huggingface.co/PygmalionAI) models to a large number of users with blazing fast speeds (thanks to vLLM's Paged Attention).

Aphrodite builds upon and integrates the exceptional work from [various projects](#acknowledgements).

The compute necessary for Aphrodite's development is provided by [Arc Compute](https://www.arccompute.io).


## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](./aphrodite/modeling/layers/attention.py) from vLLM
- Optimized CUDA kernels for improved inference
- Quantization support via AQLM, AWQ, Bitsandbytes, EXL2, GGUF, GPTQ, QuIP#, Smoothquant+, and SqueezeLLM
- Distributed inference
- Variety of sampling methods ([Mirostat](https://arxiv.org/abs/2007.14966), [Locally Typical Sampling](https://arxiv.org/abs/2202.00666), Tail-Free Sampling, etc)
- 8-bit KV Cache for higher context lengths and throughput, at both FP8 and INT8 formats.


## Quickstart

```sh
pip install aphrodite-engine

python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b
```

> [!CAUTION]
> If the installation reports CUDA kernel errors, please run `pip install aphrodite-engine=0.4.5` instead.

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports OpenAI, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).

You can play around with the engine in the demo here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlpinDale/misc-scripts/blob/main/Aphrodite.ipynb)

### Docker

Additionally, we provide a Docker image for easy deployment. Here's a basic command to get you started:

```sh
sudo docker run -d -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2" -p 2242:7860 --gpus all --ipc host alpindale/aphrodite-engine
```

This will pull the Aphrodite Engine image (~9GiB download), and launch the engine with the Mistral-7B model at port 2242. Check [here](/docker/.env) for the full list of env variables.

See [here](/docker/docker-compose.yml) for the Compose file to use with Docker Compose.

## Requirements

- Operating System: Linux (or WSL for Windows)
- Python: at least 3.8

For windows users, it's recommended to use [tabbyAPI](https://github.com/theroyallab/tabbyAPI) instead, if you do not need batching support.

#### Build Requirements:
- CUDA >= 11

For supported GPUs, see [here](https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#supported-gpus). Generally speaking, all semi-modern GPUs are supported - down to Pascal (GTX 10xx, P40, etc.)

## Installation
- [Using pip](https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#pre-compiled-binaries-via-pypi)
- [Build from source](https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#build-from-source)

## Usage

For usage, please refer to the [wiki page](https://github.com/PygmalionAI/aphrodite-engine/wiki/2.-Usage) for detailed instructions. Aphrodite provides many different options for LLM inference, so please read through the list of options [here](https://github.com/PygmalionAI/aphrodite-engine/wiki/3.-Engine-Options).

## Performance
Speeds vary with different GPUs, model sizes, quantization schemes, batch sizes, etc. Here are some baseline benchmarks conducted by requesting as many completions as possible from the [API server](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/endpoints/openai/api_server.py).

### Batch Size 1 Performance
These are the speeds a user would normally get if they request a single output with a sizable prompt and output length. Essentially, normal chatting experience.

The following results were gathered by sending a request with 8192 prompt tokens and requesting 1024 tokens with `ignore_eos=True`.

GPU: NVIDIA A40, Mistral 7B. Baseline is the same model loaded with text-generation-webui in FP16.

![](/assets/bsz1.png)

### High Batch Size Performance

> [!NOTE]  
> The numbers below are the theoretical peak achieved by *only* requesting output tokens at very high batch sizes. At lower batch sizes with much larger prompts, the results will be vastly different.
Throughput refers to output tokens per second.

This table is outdated, will be replaced soon.

| Model      | Quantization | bits | GPU      | Throughput (T/s) |
| ---------- | ------------ | ---- | -------- | ---------------- |
| Mistral 7B | None         | 16   | RTX 4090 | 5489.3           |
|            | AWQ          | 4    | RTX 4090 | 4078.8           |
|            | GPTQ         | 4    | RTX 4090 | **7850.4**       |
|            |              | 8    | RTX 4090 | 7658.0           |
|            | GGUF         | Q8   | RTX 4090 | 5141.2           |
|            |              | Q6KM | RTX 4090 | 5791.7           |
|            |              | Q5KM | RTX 4090 | 5786.2           |
|            |              | Q4KM | RTX 4090 | 5815.8           |
|            | SqueezeLLM   | 4    | RTX 4090 | 549.5            |
| Llama-2 7B | None         | 16   | RTX 4090 | 2576.2           |
|            | AWQ          | 4    | RTX 4090 | 3551.3           |
|            | GPTQ         | 4    | RTX 4090 | 2919.1           |
|            | GGUF         | Q4KM | RTX 4090 | 2726.6           |
|            |              | Q5KM | RTX 4090 | 2763.4           |
|            |              | Q6KM | RTX 4090 | 2694.7           |
|            |              | Q8   | RTX 4090 | 2647.0           |
|            | SqueezeLLM   | 4    | RTX 4090 | 580.3            |



### Notes

1. By design, Aphrodite takes up 90% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%).

2. You can view the full list of commands by running `python -m aphrodite.endpoints.openai.api_server --help`.

3. Context Length extension via the RoPE method is supported for most models. Use the command-line flag `--max-model-len` to specify a desired context length and the engine will adjust the RoPE scaling accordingly.

4. Please refer to the [FAQ & Issues](https://github.com/PygmalionAI/aphrodite-engine/wiki/6.-FAQ-&-Issues) if you run into problems. If you don't find an answer there, please make an [issue](https://github.com/PygmalionAI/aphrodite-engine/issues).

## Acknowledgements
Aphrodite Engine would have not been possible without the phenomenal work of other open-source projects. Credits go to:
- [vLLM](https://github.com/vllm-project/vllm) (CacheFlow)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [xFormers](https://github.com/facebookresearch/xformers)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/)
- [Exllamav2](https://github.com/turboderp/exllamav2)
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI)
- [AQLM](https://github.com/Vahe1994/AQLM)
- [KoboldAI](https://github.com/henk717/KoboldAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Ray](https://github.com/ray-project/ray)

## Contributing
Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
