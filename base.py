import logging
import re

import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from llama import LlamaAdapter

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

class BaseAdapter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def encode_and_decorate(self, prompt, add_special_tokens=False):
        raise NotImplementedError
    
    def decode(self, value):
        raise NotImplementedError
    
    @property
    def stopping_criteria(self):
        return None
    
    @property
    def start_ids(self):
        return [self.tokenizer.bos_token_id]
    
    @property
    def sep_ids(self):
        return [self.tokenizer.bos_token_id]
    
class BasicAdapter(BaseAdapter):
    def encode_and_decorate(self, prompt, add_special_tokens=False):
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors='pt',
        )
        logger.debug(f"Encode {prompt} to {input_ids}")
        return input_ids
    
    def decode(self, value):
        self.tokenizer: PreTrainedTokenizer
        tok = self.tokenizer.decode(value)
        return tok + ' '
    
class BasicAdapterFast(BaseAdapter):
    hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def encode_and_decorate(self, prompt, add_special_tokens=False):
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors='pt',
        )
        logger.debug(f"Encode {prompt} to {input_ids}")
        return input_ids
    
    def decode(self, value):
        self.tokenizer: PreTrainedTokenizerFast

        tok = self.tokenizer._convert_id_to_token(value)
        if tok.startswith('▁'):
            space = ' '
            tok = tok[1:]
        else:
            space = ''
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '\r':
            tok = '\n'
        
        tok = space + tok

        logger.debug(f"Decode {value} to {repr(tok)}")

        return tok
    