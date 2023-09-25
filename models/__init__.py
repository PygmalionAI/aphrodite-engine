import logging

import torch.nn as nn

from .base import BasicAdapter, BasicAdapterFast
from .llama import LlamaAdapter

logger = logging.getLogger(__name__)

def _get_default_adapter(tokenizer):
    if tokenizer.is_fast:
        return BasicAdapterFast
    else:
        return BasicAdapter
    
def init_adapter(model: nn.Module, tokenizer, adapter=None):
    if adapter is None:
        for v in model.modules():
            if 'LlamaModel' in v.__class__.__name__:
                Adapter = LlamaAdapter
                break
        else:
            Adapter = _get_default_adapter(tokenizer)
    elif adapter == 'llama1':
        Adapter = _get_default_adapter(tokenizer)
    else:
        raise ValueError(f"Adapter {adapter} is not allowed.")
    
    logger.info(f"Using adapter {Adapter.__name__}")

    return Adapter(tokenizer)