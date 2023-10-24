<h1 align="center">
Breathing Life into Language
</h1>


![aphrodite](https://raw.githubusercontent.com/PygmalionAI/aphrodite-engine/main/assets/aphrodite.png)

Aphrodite is the official backend engine for PygmalionAI. It is designed to serve as the inference endpoint for the PygmalionAI website, and to allow serving the [Pygmalion](https://huggingface.co/PygmalionAI) models to a large number of users with blazing fast speeds (thanks to FasterTransformer and vLLM). 

Aphrodite builds upon and integrates the exceptional work from [various projects](#acknowledgements).


## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](./aphrodite/modeling/layers/attention.py)
- Optimized CUDA kernels for improved inference
- Quantization support via AWQ and GPTQ
- Distributed inference
- Variety of sampling methods (top a, tail-free sampling, rep. pen.)


## Quickstart

```sh
pip install aphrodite-engine

python -m aphrodite.endpoints.api_server_kobold --model PygmalionAI/pygmalion-2-7b
```

This will create a [KoboldAI](https://github.com/henk717/KoboldAI)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports Kobold, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).

## Requirements

- Operating System: Linux (or WSL for Windows)
- Python: at least 3.8
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

Aphrodite will require a slightly specialized environment to run, as the latest CUDA versions are currently not supported. You can use Conda to easily configure your environment. If you're on windows, make sure you have [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) installed. You can do this by opening Windows PowerShell and running:
```sh
wsl --install
```

Aphrodite provides an easy-to-use install script, which helps with both setting up a suitable environment for installing via the pip package and/or building from source.

The requirements is `git`, `wget`, `bzip2`, and `tar` - all of which are available on the majority of Linux distributions, including WSL.

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine && cd aphrodite-engine
```

Then you can simply run:

```sh
./runtime.sh python -m aphrodite.endpoints.api_server_kobold --help
```

The `./runtime.sh` prefix will need to be appended to every command you run that involves Aphrodite, as it launches your commands within the created environment. If you prefer not doing that, you can run `./runtime.sh` by itself to enter the environment and execute commands as normal.

For updating the engine, run `git pull` and then `./update-runtime.sh` to update the environment.



## Usage

Aphrodite Engine provides 3 API endpoint types:

1. [KoboldAI](https://github.com/henk717/KoboldAI):
  ```sh
  python -m aphrodite.endpoints.api_server_kobold --model PygmalionAI/pygmalion-2-7b
  ```
2. [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
  ```sh
  python -m aphrodite.endpoints.api_server_ooba --model PygmalionAI/pygmalion-2-7b
  ```
3. [OpenAI](https://openai.com)
  ```sh
  python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b
  ```

Please refer to each endpoint's documentation on how to query them. Generally, they all work with [SillyTavern](https://github.com/SillyTavern/SillyTavern).

To run a quantized model, use the `--quantization` flag with either `gptq` or `awq` and the `--dtype float16` flag. Make sure your model is in AWQ/GPTQ format and not GGUF. Run with only the `--help` flag for a full list of arguments.


For the full list of Sampling parameters, please refer to [SamplingParams](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/common/sampling_params.py):

https://github.com/PygmalionAI/aphrodite-engine/blob/ab1ac578bafa922a6c7e323986bd320615311dad/aphrodite/common/sampling_params.py#L24-L88


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

2. You can view the full list of commands by running `python -m aphrodite.endpoints.api_server_ooba --help`.

3. Context Length extension via the RoPE method is supported for Llama models. Edit the `config.json` with the following values:
```json
  "rope_scaling": { "factor": 2.0, "type": "dynamic"},
```

## Acknowledgements
Aphrodite Engine would have not been possible without the phenomenal work of other open-source projects. Credits go to:
- [vLLM](https://github.com/vllm-project/vllm) (CacheFlow)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [xFormers](https://github.com/facebookresearch/xformers)
- [AWQ](https://github.com/mit-han-lab/llm-awq/)
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [ExLlama](https://github.com/turboderp/exllama)
- [KoboldAI](https://github.com/henk717/KoboldAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FastChat](https://github.com/lm-sys/FastChat)
- [SkyPilot](https://github.com/skypilot-org/skypilot)
- [OpenAI Python Library](https://github.com/openai/openai-python)

## Contributing
Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
