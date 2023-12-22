import importlib
from typing import List, Optional, Type

import torch.nn as nn

from aphrodite.common.logger import init_logger
from aphrodite.common.utils import is_hip

logger = init_logger(__name__)

# Architecture -> (module, class)
_MODELS = {
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("mistral", "MistralForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "PhiForCausalLM": ("phi_1_5", "PhiForCausalLM"),
    "YiForCausalLM": ("yi", "YiForCausalLM"),
    "LlavaForConditionalGeneration":
    ("llava", "LlavaForConditionalGeneration"),
    "LlavaMistralForCausalLM": ("bakllava", "BakLlavaForConditionalGeneration"),
}

# Models not supported by ROCm
_ROCM_UNSUPPORTED_MODELS = []

# Models partially supported by ROCm.
# Architecture -> Reason
_ROCM_PARTIALLY_SUPPORTED_MODELS = {
    "MistralForCausalLM":
    "Sliding window attention is not yet supported in ROCM's flash attention.",
    "MixtralForCausalLM":
    "Sliding window attention is not yet supported in ROCm's flash attention",
}


class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None
        if is_hip():
            if model_arch in _ROCM_UNSUPPORTED_MODELS:
                raise ValueError(f"Model architecture {model_arch} is not "
                                 "supported in ROCm for now.")
            if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    f"Model architecture {model_arch} is partially supported "
                    "by ROCm: " + _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch])

        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"aphrodite.modeling.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())


__all__ = ["ModelRegistry"]
