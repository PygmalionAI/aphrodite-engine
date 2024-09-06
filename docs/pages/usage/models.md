---
outline: deep
---

# Supported Models

Aphrodite supports a large variety of generative Transformer models in [Hugging Face Transformers](https://huggingface.co/models). The following is the list of model *architectures* that we currently support.

## Decoder-only Language Models

| Architecture            |                       Example HF Model |
| ----------------------- | -------------------------------------: |
| `AquilaForCausalLM`     |                   `BAAI/AquilaChat-7B` |
| `ArcticForCausalLM`     |  `Snowflake/snowflake-arctic-instruct` |
| `BaiChuanForCausalLM`   |      `baichuan-inc/Baichuan2-13B-Chat` |
| `BloomForCausalLM`      |                    `bigscience/bloomz` |
| `ChatGLMModel`          |                    `THUDM/chatglm3-6b` |
| `CohereForCausalLM`     |       `CohereForAI/c4ai-command-r-v01` |
| `DbrxForCausalLM`       |             `databricks/dbrx-instruct` |
| `DeciLMForCausalLM`     |                     `DeciLM/DeciLM-7B` |
| `FalconForCausalLM`     |                     `tiiuae/falcon-7b` |
| `GemmaForCausalLM`      |                      `google/gemma-7b` |
| `Gemma2ForCausalLM`     |                    `google/gemma-2-9b` |
| `GPT2LMHeadModel`       |                                 `gpt2` |
| `GPTBigCodeForCausalLM` |                    `bigcode/starcoder` |
| `GPTJForCausalLM`       |             `pygmalionai/pygmalion-6b` |
| `GPTNeoXForCausalLM`    |                `EleutherAI/pythia-12b` |
| `InternLMForCausalLM`   |                 `internlm/internlm-7b` |
| `InternLM2ForCausalLM`  |                `internlm/internlm2-7b` |
| `JAISLMHeadModel`       |                      `core42/jais-13b` |
| `JambaForCausalLM`      |                  `ai21labs/Jamba-v0.1` |
| `LlamaForCausalLM`      |         `meta-llama/Meta-Llama-3.1-8B` |
| `MiniCPMForCausalLM`    |          `openbmb/MiniCPM-2B-dpo-bf16` |
| `MistralForCausalLM`    |            `mistralai/Mistral-7B-v0.1` |
| `MixtralForCausalLM`    |          `mistralai/Mixtral-8x7B-v0.1` |
| `MPTForCausalLM`        |                      `mosaicml/mpt-7b` |
| `NemotronForCausalLM`   |              `nvidia/Minitron-8B-Base` |
| `OLMoForCausalLM`       |                   `allenai/OLMo-7B-hf` |
| `OPTForCausalLM`        |                     `facebook/opt-66b` |
| `OrionForCausalLM`      |           `OrionStarAI/Orion-14B-Chat` |
| `PhiForCausalLM`        |                      `microsoft/phi-2` |
| `Phi3ForCausalLM`       | `microsoft/Phi-3-medium-128k-instruct` |
| `Phi3SmallForCausalLM`  |  `microsoft/Phi-3-small-128k-instruct` |
| `PersimmonForCausalLM`  |              `adept/persimmon-8b-chat` |
| `QwenLMHeadModel`       |                         `Qwen/Qwen-7B` |
| `Qwen2ForCausalLM`      |                       `Qwen/Qwen2-72B` |
| `Qwen2MoeForCausalLM`   |               `Qwen/Qwen1.5-MoE-A2.7B` |
| `StableLmforCausalLM`   |         `stabilityai/stablelm-3b-4e1t` |
| `Starcoder2ForCausalLM` |                `bigcode/starcoder2-3b` |
| `XverseForCausalLM`     |               `xverse/XVERSE-65B-Chat` |

:::info
On ROCm platforms, Mistral and Mixtral are capped to 4096 max context length due to sliding window issues.
:::

## Encoder-Decoder Language Models
| Architecture                   |             Example Model |
| ------------------------------ | ------------------------: |
| `BartForConditionalGeneration` | `facebook/bart-large-cnn` |

## Multimodal Language Models

| Architecture                        | Supported Modalities |                       Example Model |
| ----------------------------------- | :------------------: | ----------------------------------: |
| `Blip2ForConditionalGeneration`     |        Image         |         `Salesforce/blip2-opt-6.7b` |
| `ChameleonForConditionalGeneration` |        Image         |             `facebook/chameleon-7b` |
| `FuyuForCausalLM`                   |        Image         |                     `adept/fuyu-8b` |
| `InternVLChatModel`                 |        Image         |            `OpenGVLab/InternVL2-8B` |
| `LlavaForConditionalGeneration`     |        Image         |         `llava-hf/llava-v1.5-7b-hf` |
| `LlavaNextForConditionalGeneration` |        Image         | `llava-hf/llava-v1.6-mistral-7b-hf` |
| `PaliGemmaForConditionalGeneration` |        Image         |        `google/paligemma-3b-pt-224` |
| `Phi3VForCausalLM`                  |        Image         | `microsoft/Phi-3.5-vision-instruct` |
| `MiniCPMV`                          |        Image         |             `openbmb/MiniCPM-V-2_6` |


If your model uses any of the architectures above, you can seamlessly run your model with Aphrodite.