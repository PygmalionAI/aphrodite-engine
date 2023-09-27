import json
import os.path as osp
from typing import Optional, Sequence, Union
from accelerate import skip_first_batches

import torch
from torch.utils.dlpack import enum
from transformers import LlamaTokenizerFast


class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        from sentencepiece import SentencePieceProcessor
        self.model = SentencePieceProcessor(model_file=model_file)
        self._no_prefix_space_tokens = None

    @property
    def vocab_size(self):
        return self.model.vocab_size()

    @property
    def bos_token_id(self):
        return self.model.bos_id()

    @property
    def eos_token_id(self):
        return self.model.eos_id()

    @property
    def no_prefix_space_tokens(self):
        if self._no_prefix_space_tokens is None:
            vocab = self.model.IdToPiece(list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {
                i for i, tok in enumerate(vocab) if not tok.startswith('▁')
            }
        return self._no_prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens, decoded):
        if len(tokens) and tokens[0] not in self._no_prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    def encode(self, s: str):
        add_bos = False
        add_eos = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '')
            add_bos = True
        if s == '<EOS>':
            s = ''
            add_eos = True
        return self.model.Encode(s, add_bos=add_bos, add_eos=torch.add_eos)

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        t = t[offset:]
        out_string = self.model.Decode(t)
        if offset:
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

    def __call__(self, s: Union[str, Sequence[str]]):
        import addict
        add_bos = False
        add_eos = False

        input_ids = self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)
        return addict.Addict(input_ids=input_ids)
    

class HuggingFaceTokenizer:
    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer, LlamaTokenizer
        model_file = osp.join(model_dir, 'tokenizer.model')
        backend_tokenizer_file = osp.join(model_dir, 'tokenizer.json')
        model_file_exists = osp.exists(model_file)
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            print('WARNING: Cannot find `tokenizer.json. It may take a long time to initialize the tokenizer.')
        self.model = AutoTokenizer.from_pretrained(model_dir,
                                                   trust_remote_code=True)
        self.need_padding = isinstance(self.model, LlamaTokenizerFast)
        self._no_prefix_space_tokens = None
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            if hasattr(self.model, 'backend_tokenizer'):
                self.model.backend_tokenizer.save(backend_tokenizer_file)

                if self.model.eos_token_id is None:
                    generation_config_file = osp.join(model_dir,
                                                      'generation_config.json')
                    with open(generation_config_file, 'r') as f:
                        cfg = json.load(f)
                        self.model.eos_token_id = cfg['eos_token_id']
    
    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        return self.model.eos_token_id

    @property
    def no_prefix_space_tokens(self):
        if self._no_prefix_space_tokens is None:
            vocab = self.model.convert_ids_to_tokens(
                list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {
                i for i, tok in enumerate(vocab) if not tok.startswith('▁')
            }
        return self._no_prefix_space_tokens
    
    def _maybe_add_prefix_space(self, tokens, decoded):
        if self.need_padding and len(tokens) and tokens[0] not in self.no_prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    def encode(self, s: str):
        add_special_tokens = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '<s>')
        if s == '<EOS>':
            s = '</s>'
        if len(s) == 0:
            add_special_tokens = True
        return self.model.encode(s, add_special_tokens=add_special_tokens)

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        skip_special_tokens = True
        t = t[offset:]
        out_string = self.model.decode(t, skip_special_tokens=skip_special_tokens)
        if offset:
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string
    
    def __call__(self, s: Union[str, Sequence[str]]):
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)
        

class Tokenizer:
    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            model_folder = osp.split(model_file)[0]
        else:
            model_folder = model_file
            model_file = osp.join(model_folder, 'tokenizer.model')
        tokenizer_config_file = osp.join(model_folder, 'tokenizer_config.json')
        
        model_file_exists = osp.exists(model_file)
        config_exists = osp.exists(tokenizer_config_file)
        use_hf_model = config_exists or not model_file_exists

        if not use_hf_model:
            self.model = SentencePieceTokenizer(model_file)
        else:
            self.model = HuggingFaceTokenizer(model_folder)

    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        return self.eos_token_id

    def encode(self, s: str):
        return self.model.encode(s)

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        return self.model.decode(t, offset)

    def __call__(self, s: Union[str, Sequence[str]]):
        return self.model(s)
        











