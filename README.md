<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](https://raw.githubusercontent.com/PygmalionAI/aphrodite-engine/main/assets/aphrodite.png)

> [!IMPORTANT]  
> Aphrodite is being actively developed on the [dev branch](https://github.com/PygmalionAI/aphrodite-engine/tree/dev). The main branch is treated as "stable".

Aphrodite is the official backend engine for PygmalionAI. It is designed to serve as the inference endpoint for the PygmalionAI website, and to allow serving the [Pygmalion](https://huggingface.co/PygmalionAI) models to a large number of users with blazing fast speeds (thanks to FasterTransformer and vLLM). 

Aphrodite builds upon and integrates the exceptional work from [various projects](#acknowledgements).


## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](./aphrodite/modeling/layers/attention.py)
- Optimized CUDA kernels for improved inference
- Quantization support via AWQ and GPTQ
- Distributed inference
- Variety of sampling methods ([Mirostat](https://arxiv.org/abs/2007.14966), [Locally Typical Sampling](https://arxiv.org/abs/2202.00666), Tail-Free Sampling, etc)


## Quickstart

```sh
pip install git+https://github.com/PygmalionAI/aphrodite-engine@dev

python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b
```

This will create a [OpenAI](https://platform.openai.com/docs/api-reference/)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports Kobold, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).


## Performance
Speeds vary with different GPUs, model sizes, quantization schemes, batch sizes, etc. Here are some baseline benchmarks conducted by requesting as many completions as possible from the [API server](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/endpoints/openai/api_server.py). Keep in mind that these are the theoritical peak throughput with parallel decoding, with as high a batch size as possible. **Per-request generation speed is a fraction of this, at 30-40 t/s**.

> [!NOTE]  
> 16bit models can achieve much higher throughput if they have access to more VRAM, either by using larger GPUs, or tensor parallelism over many GPUs.

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
- CUDA 11.8 (recommended, supports 11.0-11.8)

## Supported GPUs

Any NVIDIA GPU with a compute capability of 6.0 or higher. Refer to this page for a full list of CUDA GPUs:

[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).


Or, you can manually find out your GPU's Compute Capability by opening a Python interpreter and running:
```py
>>> import torch    # if you don't have `torch` installed, run `pip install torch` first
>>> print(torch.cuda.get_device_capability())
```
This should print something like this: `(7, 5)`, which would indicate a CC of 7.5

If you do not meet the minimum CC, you will not be able to run Aphrodite. At the moment, compute capability of 7.5 or higher is required for AWQ quantization scheme; you can use GPTQ if your GPU does not support it.

## Setting up the environment
**If you run into any problems, please refer to the common [Common Issues](#common-issues) section, or open an [Issue](https://github.com/PygmalionAI/aphrodite-engine/issues) if you can't find the answer there.**

Normally, all you need is NVIDIA device drivers. You can then simply run `pip install aphrodite-engine` to install the package. If you wish to build from source, however, then follow along with the instructions below.

Aphrodite will require a slightly specialized environment to build, as the latest CUDA versions are currently not supported. You can use Conda to easily configure your environment. If you're on windows, make sure you have [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) installed. You can do this by opening Windows PowerShell and running:
```sh
wsl --install
```

Aphrodite provides an easy-to-use install script, which helps with both setting up a suitable environment for installing via the pip package and/or building from source.

The requirements is `git`, `wget`, `bzip2`, and `tar` - all of which are available on the majority of Linux distributions. You may need to install them for WSL (`apt update && apt install git` etc).

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine && cd aphrodite-engine
```

Then you can simply run:

```sh
./runtime.sh python -m aphrodite.endpoints.openai.api_server --help
```

The `./runtime.sh` prefix will need to be appended to every command you run that involves Aphrodite, as it launches your commands within the created environment. If you prefer not doing that, you can run `./runtime.sh` by itself to enter the environment and execute commands as normal.

For updating the engine, run `git pull` and then `./update-runtime.sh` to update the environment.

Alternatively, you can install via pip:
```sh
pip install git+https://github.com/PygmalionAI/aphrodite-engine
```
Make sure you have a proper environment (with CUDA <12.0) set up.

## Usage

Aphrodite Engine provides 3 API endpoint types:

#### [OpenAI](https://openai.com)
  ```sh
  python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b --api-keys sk-example-key
  ```
Completions example:
```sh
curl http://localhost:2242/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-example-key" \
-d '{
  "model": "PygmalionAI/pygmalion-2-7b",
  "prompt": "Every age it seems is tainted by the greed of men. Rubbish to one such as I,",
  "stream": false,
  "mirostat_mode": 2,
  "mirostat_tau": 6.5,
  "mirostat_eta": 0.2
}'
```
Chat Completions example:
```sh
curl -X POST -H 'Authorization: Bearer sk-example-key' -H "Content-type: application/json" -d '{
  "model": "PygmalionAI/pygmalion-2-7b",
  "messages": [
    {
      "role": "system",
      "content": "<|system|>Enter assistant mode."
    },
    {
      "role": "user",
      "content": "Who won the world series in 2020?"
    },
    {
      "role": "assistant",
      "content": "The Los Angeles Dodgers won the World Series in 2020."
    },
    {
      "role": "user",
      "content": "Where was it played?"
    }
  ]
}' 'http://localhost:2242/v1/chat/completions'
```

For authorization, both the `Authorization: Bearer KEY` and `x-api-key: KEY` headers will work.

 #### [KoboldAI](https://github.com/henk717/KoboldAI):
```sh
python -m aphrodite.endpoints.kobold.api_server --model PygmalionAI/pygmalion-2-7b
```
cURL example:
```sh
curl -X 'POST' \
  'http://localhost:5000/api/v1/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze.",
  "max_context_length": 4096,
  "max_length": 512,
  "stream": false,
  "mirostat_mode": 2,
  "mirostat_tau": 6.5,
  "mirostat_eta": 0.2
}' 
```

To run a quantized model, use the `--quantization` flag with either `gptq`, `awq`, or `squeezellm` and the `--dtype float16` flags. Make sure your model is in the appropriate format (i.e. not GGUF or exl2). Run with only the `--help` flag for a full list of arguments.


For the full list of Sampling parameters, please refer to [SamplingParams](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/common/sampling_params.py):

https://github.com/PygmalionAI/aphrodite-engine/blob/9a47d6fc1a8ad53d6272f1ec04f848068c5c261b/aphrodite/common/sampling_params.py#L25-L104

## Common Issues
`The detected CUDA version (12.1) mismatches the version that was used to compile
      PyTorch (11.8). Please make sure to use the same CUDA versions.`

This is normally due to your environment referring to the global installation of CUDA and not the one in your current env. Run `which nvcc` and note down the output. For example, if your output is `/home/anon/miniconda3/envs/aphrodite/bin/nvcc`, run this command:
```sh
export CUDA_HOME=/home/anon/miniconda3/envs/aphrodite
```

Then run the installation command again.

***

`Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.`

You've run out of swap space! Please pass the `--swap-space` followed by the amount of swap (in GBs) to allocate. Make sure you leave enough RAM for the model loading process.

***
```
ncclInternalError: Internal check failed.
Last error:
No NVML device handle. Skipping nvlink detection.
```
This happens if you're doing tensor parallelism (multi-GPU) on NVLinked NVIDIA GPUs and they don't support P2P. Please run this command before running the server:
```sh
export NCCL_P2P_DISABLE=1
```
Alternatively, you can prepend `NCCL_P2P_DISABLE=1` to your server launch command.

### Notes

1. By design, Aphrodite takes up 90% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%).

2. You can view the full list of commands by running `python -m aphrodite.endpoints.ooba.api_server --help`.

3. Context Length extension via the RoPE method is supported for most models. Use the command-line flag `--max-model-len` to specify a desired context length and the engine will adjust the RoPE scaling accordingly.


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
