from typing import Type
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from aphrodite.common.config import ModelConfig
from aphrodite.modeling.models import LlamaForCausalLM
from aphrodite.modeling.hf_downloader import initialize_dummy_weights

_MODEL_REGISTRY = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,
}

def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are currently unsupported. "
        f"Supported architecture(s): {list(_MODEL_REGISTRY.keys())}"
    )

def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)

    # Initialize model weights as empty tensors
    model = model_class(model_config.hf_config)
    if model_config.use_dummy_weights:
        model = model.cuda()
        initialize_dummy_weights(model)
    else:
        # Load the downloaded/cached model files
        model.load_weights(
            model_config.model, model_config.download_dir,
            model_config.use_np_weights)
        model = model.cuda()
    return model.eval()