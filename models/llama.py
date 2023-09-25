import logging
import re

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from .base import BasicAdapterFast

logger = logging.getLogger(__name__)

SYS = '<|system|>'
BOT = '<|model|>'
USER = '<|user|>'
DEFAULT_SYSTEM_PROMPT = """\
Enter RP mode. You shall reply to the user while staying in character. Your responses must be detailed, creative, immersive, and drive the scenario forward. You will follow the character's persona."""

class LlamaAdapter(BasicAdapterFast):
    start_ids = []
    sep_ids = []

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.prev_round = 0

    def encode_and_decorate(self, prompt):
        if self.prev_round == 0:
            res = re.search(r'<|system|>(.*?)', prompt)
            if res:
                prompt = SYS + res.group(1).strip() + res.group(2).strip()
            else:
                prompt = SYS + DEFAULT_SYSTEM_PROMPT + prompt
            
        prompt = f'{USER}{prompt.strip()}{BOT}'

        logger.debug(f"decorated prompt: {repr(prompt)}")

        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors='pt',
        )

        self.prev_round += 1
        return input_ids
