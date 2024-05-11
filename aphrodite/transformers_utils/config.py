import os
from typing import Optional

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from loguru import logger

from aphrodite.transformers_utils.configs import (BaiChuanConfig, DbrxConfig,
                                                  ChatGLMConfig, MPTConfig,
                                                  QWenConfig, RWConfig)
from aphrodite.quantization.gguf_utils import GGUFReader

_CONFIG_REGISTRY = {
    "baichuan": BaiChuanConfig,
    "chatglm": ChatGLMConfig,
    "dbrx": DbrxConfig,
    "mpt": MPTConfig,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
}


def extract_gguf_config(checkpoint):
    if os.path.isfile(checkpoint):
        result = GGUFReader(checkpoint)
    elif os.path.isdir(checkpoint):
        try:
            return AutoConfig.from_pretrained(checkpoint)
        except Exception:
            pass

        all_gguf_files = sorted([
            file for file in os.listdir(checkpoint)
            if os.path.splitext(file)[-1].lower() == ".gguf"
        ])
        # assume the config is always in the first shard
        result = GGUFReader(os.path.join(checkpoint, all_gguf_files[0]))
    else:
        raise RuntimeError(f"Cannot find any model config with `{checkpoint}`")

    logger.info("Extracting config from GGUF...")
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]),
                       encoding='utf-8')
    # Only support llama so far
    if architecture != "llama":
        raise RuntimeError(f"Unsupported architecture {architecture}, "
                           "only llama is supported.")

    # write config
    vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
    context_length = int(result.fields['llama.context_length'].parts[-1])
    n_layer = int(result.fields['llama.block_count'].parts[-1])
    n_head = int(result.fields['llama.attention.head_count'].parts[-1])
    n_local_heads = int(
        result.fields['llama.attention.head_count_kv'].parts[-1])
    intermediate_size = int(
        result.fields['llama.feed_forward_length'].parts[-1])
    norm_eps = float(
        result.fields['llama.attention.layer_norm_rms_epsilon'].parts[-1])
    dim = int(result.fields['llama.embedding_length'].parts[-1])
    arch = "MixtralForCausalLM"
    if 'llama.expert_count' in result.fields:
        arch = "MixtralForCausalLM"
        name = "mixtral"
    else:
        arch = "LlamaForCausalLM"
        name = "llama"
    model_config = {
        "architectures": [arch],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": dim,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": context_length,
        "model_type": name,
        "num_attention_heads": n_head,
        "num_hidden_layers": n_layer,
        "num_key_value_heads": n_local_heads,
        "rms_norm_eps": norm_eps,
        "torch_dtype": "float16",
        "vocab_size": vocab_size
    }
    if 'llama.rope.freq_base' in result.fields:
        model_config['rope_theta'] = float(
            result.fields['llama.rope.freq_base'].parts[-1])
    if 'llama.expert_count' in result.fields:
        model_config['num_local_experts'] = int(
            result.fields['llama.expert_count'].parts[-1])
        model_config['num_experts_per_tok'] = int(
            result.fields['llama.expert_used_count'].parts[-1])
    if name in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[name]
    else:
        config_class = CONFIG_MAPPING[name]
    hf_config = config_class.from_dict(model_config)
    return hf_config


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None,
               code_revision: Optional[str] = None) -> PretrainedConfig:
    if model.endswith("gguf"):
        return extract_gguf_config(model)
    try:
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model,
                                              revision=revision,
                                              code_revision=code_revision)
    return config


def get_hf_text_config(config: PretrainedConfig):
    """Get the `sub` config relevant to multimodal models.
    No-op for text models.
    """
    if hasattr(config, "text_config"):
        # The code operates under the assumption that
        # text_config should have `num_attention_heads`
        # (among others). Assert here to fail early
        # if transformer config doesn't align with
        # the assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    else:
        return config
