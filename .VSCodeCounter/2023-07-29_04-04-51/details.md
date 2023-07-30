# Details

Date : 2023-07-29 04:04:51

Directory /home/alpindale/AI-Stuff/projects/aphrodite-engine/aphrodite

Total : 52 files,  5352 codes, 1197 comments, 1094 blanks, all 7643 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [aphrodite/__init__.py](/aphrodite/__init__.py) | Python | 19 | 0 | 3 | 22 |
| [aphrodite/common/__init__.py](/aphrodite/common/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/common/block.py](/aphrodite/common/block.py) | Python | 44 | 7 | 17 | 68 |
| [aphrodite/common/config.py](/aphrodite/common/config.py) | Python | 171 | 51 | 33 | 255 |
| [aphrodite/common/logger.py](/aphrodite/common/logger.py) | Python | 29 | 9 | 15 | 53 |
| [aphrodite/common/outputs.py](/aphrodite/common/outputs.py) | Python | 72 | 23 | 14 | 109 |
| [aphrodite/common/sampling_params.py](/aphrodite/common/sampling_params.py) | Python | 97 | 37 | 10 | 144 |
| [aphrodite/common/sequence.py](/aphrodite/common/sequence.py) | Python | 199 | 50 | 48 | 297 |
| [aphrodite/common/utils.py](/aphrodite/common/utils.py) | Python | 26 | 3 | 10 | 39 |
| [aphrodite/endpoints/__init__.py](/aphrodite/endpoints/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/endpoints/api_server.py](/aphrodite/endpoints/api_server.py) | Python | 58 | 8 | 13 | 79 |
| [aphrodite/endpoints/llm.py](/aphrodite/endpoints/llm.py) | Python | 96 | 48 | 13 | 157 |
| [aphrodite/endpoints/openai/__init__.py](/aphrodite/endpoints/openai/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/endpoints/openai/api_server.py](/aphrodite/endpoints/openai/api_server.py) | Python | 465 | 55 | 74 | 594 |
| [aphrodite/endpoints/openai/protocol.py](/aphrodite/endpoints/openai/protocol.py) | Python | 130 | 1 | 30 | 161 |
| [aphrodite/engine/__init__.py](/aphrodite/engine/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/engine/aphrodite_engine.py](/aphrodite/engine/aphrodite_engine.py) | Python | 250 | 62 | 50 | 362 |
| [aphrodite/engine/args_tools.py](/aphrodite/engine/args_tools.py) | Python | 88 | 5 | 12 | 105 |
| [aphrodite/engine/async_aphrodite.py](/aphrodite/engine/async_aphrodite.py) | Python | 135 | 45 | 29 | 209 |
| [aphrodite/engine/ray_tools.py](/aphrodite/engine/ray_tools.py) | Python | 62 | 21 | 14 | 97 |
| [aphrodite/modeling/__init__.py](/aphrodite/modeling/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [aphrodite/modeling/hf_downloader.py](/aphrodite/modeling/hf_downloader.py) | Python | 94 | 13 | 16 | 123 |
| [aphrodite/modeling/layers/__init__.py](/aphrodite/modeling/layers/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/modeling/layers/activation.py](/aphrodite/modeling/layers/activation.py) | Python | 27 | 11 | 7 | 45 |
| [aphrodite/modeling/layers/attention.py](/aphrodite/modeling/layers/attention.py) | Python | 250 | 132 | 35 | 417 |
| [aphrodite/modeling/layers/layernorm.py](/aphrodite/modeling/layers/layernorm.py) | Python | 21 | 6 | 4 | 31 |
| [aphrodite/modeling/layers/sampler.py](/aphrodite/modeling/layers/sampler.py) | Python | 306 | 42 | 56 | 404 |
| [aphrodite/modeling/loader.py](/aphrodite/modeling/loader.py) | Python | 35 | 2 | 5 | 42 |
| [aphrodite/modeling/megatron/README.md](/aphrodite/modeling/megatron/README.md) | Markdown | 1 | 0 | 0 | 1 |
| [aphrodite/modeling/megatron/__init__.py](/aphrodite/modeling/megatron/__init__.py) | Python | 6 | 0 | 2 | 8 |
| [aphrodite/modeling/megatron/parallel_state.py](/aphrodite/modeling/megatron/parallel_state.py) | Python | 369 | 100 | 103 | 572 |
| [aphrodite/modeling/megatron/tensor_parallel/__init__.py](/aphrodite/modeling/megatron/tensor_parallel/__init__.py) | Python | 40 | 5 | 5 | 50 |
| [aphrodite/modeling/megatron/tensor_parallel/layers.py](/aphrodite/modeling/megatron/tensor_parallel/layers.py) | Python | 265 | 135 | 50 | 450 |
| [aphrodite/modeling/megatron/tensor_parallel/mappings.py](/aphrodite/modeling/megatron/tensor_parallel/mappings.py) | Python | 156 | 38 | 89 | 283 |
| [aphrodite/modeling/megatron/tensor_parallel/random.py](/aphrodite/modeling/megatron/tensor_parallel/random.py) | Python | 73 | 67 | 26 | 166 |
| [aphrodite/modeling/megatron/tensor_parallel/utils.py](/aphrodite/modeling/megatron/tensor_parallel/utils.py) | Python | 34 | 26 | 12 | 72 |
| [aphrodite/modeling/metadata.py](/aphrodite/modeling/metadata.py) | Python | 46 | 13 | 6 | 65 |
| [aphrodite/modeling/models/__init__.py](/aphrodite/modeling/models/__init__.py) | Python | 8 | 0 | 1 | 9 |
| [aphrodite/modeling/models/gpt_j.py](/aphrodite/modeling/models/gpt_j.py) | Python | 197 | 22 | 31 | 250 |
| [aphrodite/modeling/models/gpt_neox.py](/aphrodite/modeling/models/gpt_neox.py) | Python | 202 | 6 | 30 | 238 |
| [aphrodite/modeling/models/llama.py](/aphrodite/modeling/models/llama.py) | Python | 261 | 25 | 35 | 321 |
| [aphrodite/modeling/utils.py](/aphrodite/modeling/utils.py) | Python | 13 | 1 | 3 | 17 |
| [aphrodite/processing/__init__.py](/aphrodite/processing/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/processing/block_manager.py](/aphrodite/processing/block_manager.py) | Python | 175 | 21 | 36 | 232 |
| [aphrodite/processing/policy.py](/aphrodite/processing/policy.py) | Python | 33 | 0 | 9 | 42 |
| [aphrodite/processing/scheduler.py](/aphrodite/processing/scheduler.py) | Python | 316 | 60 | 51 | 427 |
| [aphrodite/task_handler/__init__.py](/aphrodite/task_handler/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/task_handler/cache_engine.py](/aphrodite/task_handler/cache_engine.py) | Python | 122 | 9 | 24 | 155 |
| [aphrodite/task_handler/worker.py](/aphrodite/task_handler/worker.py) | Python | 251 | 17 | 51 | 319 |
| [aphrodite/transformers_utils/__init__.py](/aphrodite/transformers_utils/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [aphrodite/transformers_utils/config.py](/aphrodite/transformers_utils/config.py) | Python | 17 | 0 | 1 | 18 |
| [aphrodite/transformers_utils/tokenizer.py](/aphrodite/transformers_utils/tokenizer.py) | Python | 85 | 21 | 12 | 118 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)