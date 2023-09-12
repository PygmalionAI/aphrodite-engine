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

:warning:
**Please note that Aphrodite only supports 16-bit HuggingFace models (no GGML or GPTQ). Please refer to the notes below for important information.**

## Features

- Continuous Batching
- Efficient K/V management with [PagedAttention](./aphrodite/modeling/layers/attention.py)
- Optimized CUDA kernels for improved inference
- Distributed inference
- Multiple decoding algorithms (e.g. parallel sampling, beam search)


## Requirements

- Operating System: Linux (or WSL for Windows)
- Python: at least 3.8
- CUDA 11.7 (recommended, supports 11.0-11.8)

## Supported GPUs

Basically, anything with a compute capability of 6.0 or higher. Refer to this page for a full list of CUDA GPUs:

[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).


Or, you can manually find out your GPU's Compute Capability by opening a Python interpreter and running:
```py
>>> import torch    # if you don't have `torch` installed, run `pip install torch` first
>>> print(torch.cuda.get_device_capability())
```
This should print something like this: `(7, 5)`, which would indicate a CC of 7.5

If you do not meet the minimum CC, you will not be able to run Aphrodite.

## Setting up the environment
:grey_exclamation:
**If you run into any problems, please refer to the common [Common Issues](#common-issues) section, or open an [Issue](https://github.com/PygmalionAI/aphrodite-engine/issues) if you can't find the answer there.**

Aphrodite will require a slightly specialized environment to run, as the latest CUDA and GCC versions are not supported. You can use Conda to easily configure your environment.

### Install miniconda3

```sh
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash ./Miniconda3*
```
You can follow the on-screen instructions, though you may want to set the installation directory to somewhere with a large empty storage space.

You can either source your shell script (`. ~/.bashrc` or `. ~/.zshrc`) or restart your terminal instance to begin using conda.

### Configuring the env for Aphrodite-engine
```sh
$ conda config --set auto_activate_base false
$ conda create -n aphrodite python=3.10
$ conda activate aphrodite
$ conda install -c conda-forge cudatoolkit-dev gcc=11.3 gxx=11.3
```
:warning: If you're using an NVIDIA H100 card, please run these install commands instead:
```sh
$ sudo apt-get install ninja-build -y
$ conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc=11.8
$ pip install git+https://github.com/facebookresearch/xformers.git
```

The last command will take a long time, depending on your internet speed and processor count.

Whenever you want to launch Aphrodite later on, make sure you run `conda activate aphrodite` first. The other steps outlined above are one-time only.

## Installation
- Clone the repository:
  ```sh
  git clone https://github.com/PygmalionAI/aphrodite-engine && cd aphrodite-engine
  ```
- Install the package:
  ```
  pip install -e .
  ```
  > If you receive any import errors here, try running `pip install -r requirements.txt` first.


## Example usage

### Inference with `LLM`
  ```py
  from aphrodite import LLM, SamplingParams

  prompts = [
    "What is a man? A",
    "The sun is a wondrous body, like a magnificent",
    "All flesh is grass and all the comeliness thereof",
  ]
  sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

  llm = LLM(model="EleutherAI/pythia-70m")        # you can also use a local directory path
  outputs = llm.generate(prompts, sampling_params)
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
  ```

### Continuous inference with API
```sh
$ python -m aphrodite.endpoints.openai.api_server --model EleutherAI/pythia-70m
$ curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "EleutherAI/pythia-70m",
        "prompt": "What is a man? A",
        "max_tokens": 512,
        "temperature": 0.8
    }'
```

#### Chat API
```sh
$ curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "Act out the scenario below in a fictional setting."
      },
      { "role": "assistant", "content": "[First Message]" },
      { "role": "user", "content": "---user input---" },
    ],
    "model": "EleutherAI/pythia-70m",
    "temperature": 0.9,
    "max_tokens": 500,
    "stream": false,
    "presence_penalty": 0.7,
    "frequency_penalty": 0.7,
    "top_p": 1,
    "top_k": -1,
    "logit_bias": {}
  }'
```
For the full list of request parameters, see [OpenAI Completions API reference](https://platform.openai.com/docs/api-reference/completions).

### Common Issues
- `The detected CUDA version (12.1) mismatches the version that was used to compile
      PyTorch (11.7). Please make sure to use the same CUDA versions.`

This is normally due to your environment referring to the global installation of CUDA and not the one in your current env. Run `which nvcc` and note down the output. For example, if your output is `/home/anon/miniconda3/envs/aphrodite/bin/nvcc`, run this command:
```sh
$ export CUDA_HOME=/home/anon/miniconda3/envs/aphrodite
```

Then run the installation command again.

- `Cuda failure 'peer access is not supported between these two devices' [repeated 15x across cluster]`

  
This would be the last line in a very long error message. This happens if you're using a cluster of NVLinked GPUs and (possibly) using more than 2 of them at once. To fix this, run these two before starting the engine:

```sh
$ export NCCL_IGNORE_DISABLED_P2P=1
$ export NCCL_P2P_DISABLE=1
```

- `Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.`

You've run out of swap space! Please pass the `--swap-space` followed by the amount of swap (in GBs) to allocate. Make sure you leave enough RAM for the model loading process.
### Notes

1. Currently, only FP16/BF16 precision HuggingFace models are supported. These are the default model types you might find on a HuggingFace upload. GPTQ and GGML are **not** supported.

2. By design, Aphrodite takes up 88% of your GPU's VRAM. If you're not serving an LLM at scale, you may want to limit the amount of memory it takes up. You can do this in the API example by launching the server with the `--gpu-memory-utilization 0.6` (0.6 means 60%).

3. You can view the full list of commands by running `python -m aphrodite.endpoints.openai.api_server --help`.

## Contributing
We accept PRs! There will likely be a few typos or other errors we've failed to catch, so please let us know either via an issue or by making a Pull Request.
