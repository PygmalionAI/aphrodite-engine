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
- Variety of sampling methods ([Mirostat](https://arxiv.org/abs/2007.14966), [Locally Typical Sampling](https://arxiv.org/abs/2202.00666), Tail-Free Sampling, etc)


## Quickstart

```sh
pip install aphrodite-engine

python -m aphrodite.endpoints.kobold.api_server --model PygmalionAI/pygmalion-2-7b
```

This will create a [KoboldAI](https://github.com/henk717/KoboldAI)-compatible API server that can be accessed at port 2242 of the localhost. You can plug in the API into a UI that supports Kobold, such as [SillyTavern](https://github.com/SillyTavern/SillyTavern).


## Performance
Speeds vary with different GPUs, model sizes, quantization schemes, batch sizes, etc. Here are some baseline benchmarks conducted by sending requests of varying lengths to the provided [API server](https://github.com/PygmalionAI/aphrodite-engine/blob/main/aphrodite/endpoints/ooba/api_server.py).

| Model | Quantization | GPU      | Request Rate | Throughput (req/s) | Avg Latency (s) |
| ----- | ------------ | -------- | ------------ | ------------------ | --------------- |
| 7B    | None         | RTX 3090 | 19           | **2.66**           | **18.38**       |
| 7B    | AWQ          | RTX 3090 | 12           | **3.08**           | **32.47**       |
| 7B    | GPTQ         | RTX 3090 | 12           | **2.01**           | **49.78**       |
| 13B   | AWQ          | RTX 3090 | 5            | **1.77**           | **26.77**       |
| 13B   | GPTQ         | RTX 3090 | 5            | **1.10**           | **39.80**       |
| 20B   | AWQ          | RTX 3090 | 3            | **0.94**           | **39.07**       |
| 20B   | GPTQ         | RTX 3090 | 3            | **0.58**           | **75.54**       |

Benchmarks with other GPUs will be added soon.
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

The requirements is `git`, `wget`, `bzip2`, and `tar` - all of which are available on the majority of Linux distributions. You may need to install them for WSL (`apt update && apt install git` etc).

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine && cd aphrodite-engine
```

Then you can simply run:

```sh
./runtime.sh python -m aphrodite.endpoints.kobold.api_server --help
```

The `./runtime.sh` prefix will need to be appended to every command you run that involves Aphrodite, as it launches your commands within the created environment. If you prefer not doing that, you can run `./runtime.sh` by itself to enter the environment and execute commands as normal.

For updating the engine, run `git pull` and then `./update-runtime.sh` to update the environment.

Note that the command above builds the engine from source, which may take up to 10 minutes. Alternatively, you can install via pip:
```sh
pip install aphrodite-engine
```
Make sure you have a proper environment (with CUDA >12.0) set up.

## Usage

Aphrodite Engine provides 3 API endpoint types:

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
  "stream": false,
  "mirostat_mode": 2,
  "mirostat_tau": 6.5,
  "mirostat_eta": 0.2
}' 
```


#### [Text Generation WebUI (legacy)](https://github.com/oobabooga/text-generation-webui)
  ```sh
  python -m aphrodite.endpoints.ooba.api_server --model PygmalionAI/pygmalion-2-7b --api-keys EMPTY
  ```
cURL example:
```sh
curl -X POST "http://localhost:2242/api/v1/generate" \
-H "Content-Type: application/json" \
-H "x-api-key: EMPTY" \
-d '{
  "prompt": "This is a cake recipe:\n\n1.",
  "stream": false,
  "mirostat_mode": 2,
  "mirostat_tau": 6.5,
  "mirostat_eta": 0.2
}'
```

#### [OpenAI](https://openai.com)
  ```sh
  python -m aphrodite.endpoints.openai.api_server --model PygmalionAI/pygmalion-2-7b --api-keys EMPTY
  ```
cURL Completions example:
```sh
curl http://localhost:2242/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPTY" \
-d '{
  "model": "PygmalionAI/pygmalion-2-7b",
  "prompt": "Every age it seems is tainted by the greed of men. Rubbish to one such as I,",
  "stream": false,
  "mirostat_mode": 2,
  "mirostat_tau": 6.5,
  "mirostat_eta": 0.2
}'
```
cURL Chat Completions example:
```sh
curl -X POST -H 'Authorization: Bearer EMPTY' -H "Content-type: application/json" -d '{
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

For authorization, both the `Authorization: Bearer KEY` and `x-api-key: KEY` will work.


To run a quantized model, use the `--quantization` flag with either `gptq` or `awq` and the `--dtype float16` flag. Make sure your model is in AWQ/GPTQ format and not GGUF. Run with only the `--help` flag for a full list of arguments.


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

3. Context Length extension via the RoPE method is supported for Llama models. Edit the `config.json` with the following values or pass the desired context length size to the `--max-model-len` arg:
```json
  "rope_scaling": { "factor": 2.0, "type": "dynamic"},
```


## Acknowledgements
Aphrodite Engine would have not been possible without the phenomenal work of other open-source projects. Credits go to:
- [vLLM](https://github.com/vllm-project/vllm) (CacheFlow)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [xFormers](https://github.com/facebookresearch/xformers)
- [AWQ](https://github.com/casper-hansen/AutoAWQ)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [ExLlama](https://github.com/turboderp/exllama)
- [Exllamav2](https://github.com/turboderp/exllamav2)
- [KoboldAI](https://github.com/henk717/KoboldAI)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FastChat](https://github.com/lm-sys/FastChat)
- [SkyPilot](https://github.com/skypilot-org/skypilot)
- [OpenAI Python Library](https://github.com/openai/openai-python)

## Contributing
Everyone is welcome to contribute. You can support the project by opening Pull Requests for new features, fixes, or general UX improvements.
