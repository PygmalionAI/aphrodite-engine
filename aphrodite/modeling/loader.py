"""Utilities for selecting and loading models."""
import contextlib
import gc
from contextlib import nullcontext
from typing import Type
from loguru import logger

import torch
import torch.nn as nn

from aphrodite.common.config import DeviceConfig, ModelConfig
from aphrodite.modeling.models import ModelRegistry
from aphrodite.modeling.models.llava import LlavaForConditionalGeneration
from aphrodite.modeling.hf_downloader import (
    get_quant_config,
    initialize_dummy_weights,
)
from aphrodite.quantization.bitsandbytes import (
    BNBLinearMethod,
    replace_quant_params,
)
from aphrodite.distributed import (
    get_tensor_model_parallel_world_size, )

_VISION_MODEL_CLASSES = [
    LlavaForConditionalGeneration,
]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(model_config: ModelConfig) -> Type[nn.Module]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> nn.Module:
    lora_config = kwargs.get("lora_config", None)
    vision_language_config = kwargs.get("vision_language_config", None)
    model_class = _get_model_architecture(model_config)

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            # set the dtype to float16 for quantized models
            model_config.dtype = torch.float16
            logger.warning("Model is quantized. Forcing float16 datatype.")
        linear_method = quant_config.get_linear_method()

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device(device_config.device) if not (
                isinstance(linear_method, BNBLinearMethod)
                and linear_method.quant_config.from_float) else nullcontext():
            if hasattr(model_class, "supported_lora_modules"):
                model = model_class(model_config.hf_config, linear_method,
                                    lora_config)
            elif lora_config:
                raise ValueError(
                    f"Model {model_class.__name__} does not support LoRA, "
                    "but LoRA is enabled. Support for this model may "
                    "be added in the future. If this is important to you, "
                    "please open an issue on github.")
            else:
                if model_class not in _VISION_MODEL_CLASSES:
                    model = model_class(model_config.hf_config, linear_method)
                else:
                    model = model_class(model_config.hf_config,
                                        vision_language_config, linear_method)
        if model_config.load_format == "dummy":
            # NOTE: For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)

        if isinstance(linear_method, BNBLinearMethod):
            replace_quant_params(
                model,
                quant_config=linear_method.quant_config,
                modules_to_not_convert="lm_head",
            )
            torch.cuda.synchronize()
            if linear_method.quant_config.from_float:
                model = model.cuda()
            gc.collect()
            torch.cuda.empty_cache()
            tp = get_tensor_model_parallel_world_size()
            logger.info(
                "Memory allocated for converted model: {} GiB x {} = {} "
                "GiB".format(
                    round(
                        torch.cuda.memory_allocated(
                            torch.cuda.current_device()) /
                        (1024 * 1024 * 1024),
                        2,
                    ),
                    tp,
                    round(
                        torch.cuda.memory_allocated(
                            torch.cuda.current_device()) * tp /
                        (1024 * 1024 * 1024),
                        2,
                    ),
                ))
            logger.info(
                "Memory reserved for converted model: {} GiB x {} = {} "
                "GiB".format(
                    round(
                        torch.cuda.memory_reserved(torch.cuda.current_device())
                        / (1024 * 1024 * 1024),
                        2,
                    ),
                    tp,
                    round(
                        torch.cuda.memory_reserved(torch.cuda.current_device())
                        * tp / (1024 * 1024 * 1024),
                        2,
                    ),
                ))
    return model.eval()
