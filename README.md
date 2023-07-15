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

- Operating System: Linux (or WSL for Windows)
- Python: at least 3.8
- CUDA 11.7 (recommended, supports 11.0-11.8)

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

> \* CC: Compute Capability

Most datacenter/workstation GPUs are supported, so long as they have a compute capability of 7.0 or higher.

If you're unsure, you can find out by opening a Python interpreter and running:
```py
>>> import torch
>>> print(torch.cuda.get_device_capability())
```
This should print something like this: `(7, 5)`, which would indicate a CC of 7.5

If your GPU is not listed here or you do not meet the minimum CC, you will not be able to run Aphrodite.

## Setting up the environment

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
$ conda create -n aphrodite python=3.9
$ conda activate aphrodite
$ conda install -c conda-forge cudatoolkit-dev gcc=11.3 gxx=11.3
```
The last command will take a long time, depending on your internet speed.

Whenever you want to launch Aphrodite later on, make sure you run `conda activate aphrodite` first. The other steps outlined above are one-time only.

## Insallation
- Clone the repository:
  ```sh
  git clone https://github.com/PygmalionAI/aphrodite-engine && cd aphrodite-engine
  ```
- Install the package:
  ```
  pip install -e .
  ```
  > If you receive any import errors here, try running `pip install -r requirements.txt` first.

**If you receive an error for CUDA version mismatch**, run `which nvcc` and note down the output. For example, if your output is `/home/anon/miniconda3/envs/aphrodite/bin/nvcc`, run this command:
```sh
$ export CUDA_HOME=/home/anon/miniconda3/envs/aphrodite
```
Then run the installation command again.

## Example usage
**Currently not working, but this is how you'd run it once it's fixed.**
  ```py
  from aphrodite import LLM, SamplingParams

  prompts = [
    "What is a man? A",
    "The sun is a wondrous body, like a magnificent",
    "All flesh is grass and all the comeliness thereof",
  ]
  sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

  llm = LLM(model="EleutherAI/pythia-70m")
  outputs = llm.generate(prompts, sampling_params)
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
  ```


## Contributing
We accept PRs! There will likely be a few typos or other errors we've failed to catch, so please let us know either via an issue or make a Pull Request.
