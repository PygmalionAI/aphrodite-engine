<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](https://raw.githubusercontent.com/PygmalionAI/aphrodite-engine/main/assets/aphrodite.png)

Aphrodite is the official backend engine for PygmalionAI. It is designed to serve as the inference endpoint for the PygmalionAI website, and to allow serving the [Pygmalion](https://huggingface.co/PygmalionAI) models to a large number of users with blazing fast speeds (thanks to FasterTransformer and vLLM). 

Aphrodite builds upon and integrates the exceptional work from [various projects](#acknowledgements).

The compute necessary for Aphrodite's development is provided by [Arc Compute](https://www.arccompute.io).


## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](./aphrodite/modeling/layers/attention.py)
- Optimized CUDA kernels for improved inference
- Quantization support via GPTQ, GGUF, AWQ, QuIP#, and SqueezeLLM.
- Distributed inference
- Variety of sampling methods ([Mirostat](https://arxiv.org/abs/2007.14966), [Locally Typical Sampling](https://arxiv.org/abs/2202.00666), Tail-Free Sampling, etc)
- 8-bit KV Cache for higher context lengths and throughput.


## Quickstart

```sh
pip install aphrodite-engine

python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b
```

> [!CAUTION]
> If the installation reports CUDA kernel errors, please run `pip install aphrodite-engine=0.4.5` instead.

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports Kobold, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).

### Docker
Additionally, we provide a docker image for easy deployment. Here's a base command to get you started:
```
sudo docker run --gpus '"all"' --shm-size 10g -p 2242:2242 -it alpindale/aphrodite-engine
```

This will pull the Aphrodite Engine image (~9GiB download), and throw you in a bash commandline. From there, follow the instructions [here](https://github.com/PygmalionAI/aphrodite-engine/wiki/2.-Usage) to
create an OpenAI-compatible API.

## Performance
Speeds vary with different GPUs, model sizes, quantization schemes, batch sizes, etc. Here are some baseline benchmarks conducted by requesting as many completions as possible from the [API server](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/endpoints/openai/api_server.py). Keep in mind that these are the theoritical peak throughput with parallel decoding, with as high a batch size as possible. **Per-request generation speed is a fraction of this, at 30-40 t/s**.

### High Batch Size Performance

> [!NOTE]  
> The numbers below are the theoritical peak achieved by *only* requesting output tokens at very high batch sizes. At lower batch sizes with much larger prompts, the results will be vastly different.
Throughput refers to output tokens per second.

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

### Batch Size 1
These are the speeds a user would normally get if they request a single output with a sizable prompt and output length. Essentially, normal chatting experience.

The following results were gathered by sending a request with 2000 prompt tokens and requesting 1024 tokens with `ignore_eos=True`.

| Model      | Quantization | bits | GPU      | Throughput (T/s) |
| ---------- | ------------ | ---- | -------- | ---------------- |
| Mistral 7B | None         | 16   | RTX 4090 | 54.0             |
|            | AWQ          | 4    | RTX 4090 | 128.2            |
|            | GPTQ         | 8    | RTX 4090 | 92.8             |
|            |              | 4    | RTX 4090 | **146.8**        |
|            | GGUF         | Q8   | RTX 4090 | 91.0             |
|            |              | Q6KM | RTX 4090 | 105.4            |
|            |              | Q5KM | RTX 4090 | 117.8            |
|            |              | Q4KM | RTX 4090 | 128.9            |
| Llama-2 7B | None         | 16   | RTX 4090 | 55.2             |
|            | GPTQ         | 8    | RTX 4090 | 90.2             |
|            |              | 4    | RTX 4090 | **128.0**        |
|            | AWQ          | 4    | RTX 4090 | 116.3            |
|            | GGUF         | Q8   | RTX 4090 | 88.1             |
|            |              | Q6KM | RTX 4090 | 99.4             |
|            |              | Q5KM | RTX 4090 | 109.9            |
|            |              | Q4KM | RTX 4090 | 118.9            |

## Requirements

- Operating System: Linux (or WSL for Windows)
- Python: at least 3.8

#### Build Requirements:
- CUDA >=12

For supported GPUs, see [here](https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#supported-gpus).

## Installation
- [From PyPi](https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#pre-compiled-binaries-via-pypi)
- [Build from source](https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#build-from-source)

## Usage

For usage, please refer to the [wiki page](https://github.com/PygmalionAI/aphrodite-engine/wiki/2.-Usage) for detailed instructions. Aphrodite provides many different options for LLM inference, so please read through the list of options [here](https://github.com/PygmalionAI/aphrodite-engine/wiki/3.-Engine-Options).

### Notes

1. By design, Aphrodite takes up 90% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%).

2. You can view the full list of commands by running `python -m aphrodite.endpoints.openai.api_server --help`.

3. Context Length extension via the RoPE method is supported for most models. Use the command-line flag `--max-model-len` to specify a desired context length and the engine will adjust the RoPE scaling accordingly.

4. Please refer to the [FAQ & Issues](https://github.com/PygmalionAI/aphrodite-engine/wiki/6.-FAQ-&-Issues) if you run into problems. If you don't find an answer there, please make an [issue](https://github.com/PygmalionAI/aphrodite-engine/issues).

## Acknowledgements
Aphrodite Engine would have not been possible without the phenomenal work of other open-source projects. Credits go to:
- [vLLM](https://github.com/vllm-project/vllm) (CacheFlow)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [xFormers](https://github.com/facebookresearch/xformers)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM/)
- [ExLlama](https://github.com/turboderp/exllama)
- [Exllamav2](https://github.com/turboderp/exllamav2)
- [KoboldAI](https://github.com/henk717/KoboldAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FastChat](https://github.com/lm-sys/FastChat)
- [Ray](https://github.com/ray-project/ray)
- [SkyPilot](https://github.com/skypilot-org/skypilot)
- [OpenAI Python Library](https://github.com/openai/openai-python)

## Contributing
Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
