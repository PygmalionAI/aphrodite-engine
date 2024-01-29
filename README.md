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
- Quantization support via GPTQ, AWQ, and SqueezeLLM.
- Distributed inference
- Variety of sampling methods ([Mirostat](https://arxiv.org/abs/2007.14966), [Locally Typical Sampling](https://arxiv.org/abs/2202.00666), Tail-Free Sampling, etc)
- 8-bit KV Cache for higher context lengths and throughput.


## Quickstart

```sh
pip install aphrodite-engine

python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b
```

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports Kobold, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).


## Performance
Speeds vary with different GPUs, model sizes, quantization schemes, batch sizes, etc. Here are some baseline benchmarks conducted by requesting as many completions as possible from the [API server](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/endpoints/openai/api_server.py). Keep in mind that these are the theoritical peak throughput with parallel decoding, with as high a batch size as possible. **Per-request generation speed is a fraction of this, at 30-40 t/s**.

> [!NOTE]  
> 16bit models can achieve much higher throughput if they have access to more VRAM, either by using larger GPUs, or tensor parallelism over many GPUs. The numbers below are purely for output tokens.

| Model      | Quantization | GPU      | Throughput (output t/s) |
| ---------- | ------------ | -------- | ----------------------- |
| Llama-2 7B | None         | RTX 4090 | 2576.2                  |
|            | AWQ          | RTX 4090 | 3551.3                  |
|            | GPTQ         | RTX 4090 | 2919.1                  |
|            | SqueezeLLM   | RTX 4090 | 580.3                   |
| Mistral 7B | None         | RTX 4090 | 5489.3                  |
|            | AWQ          | RTX 4090 | 4078.8                  |
|            | GPTQ         | RTX 4090 | 4516.2                  |
|            | SqueezeLLM   | RTX 4090 | 549.5                   |

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
