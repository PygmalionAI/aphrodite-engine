from typing import Optional
from transformers import AutoConfig, PretrainedConfig
from aphrodite.transformers_utils.configs import * 

def get_config(model: str, trust_remote_code: bool,
               revision: Optional[str] = None) -> PretrainedConfig:
    if "mistral" in model.lower():
        return MistralConfig.from_pretrained(model, revision=revision)
    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model uses custom "
                "code not yet available in HF transformers library, consider "
                "setting `trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return config
