import os
import tempfile
from typing import List, Optional, Tuple, Union

import gguf
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, LlamaTokenizer)
from transformers.convert_slow_tokenizer import import_protobuf
from loguru import logger

from aphrodite.lora.request import LoRARequest
from aphrodite.common.utils import make_async, LRUCache
from aphrodite.transformers_utils.tokenizers import BaichuanTokenizer


def convert_gguf_to_tokenizer(checkpoint):
    result = gguf.GGUFReader(checkpoint)
    # write vocab
    sentencepiece_model_pb2 = import_protobuf()
    vocab = sentencepiece_model_pb2.ModelProto()
    vocab_size = len(result.fields['tokenizer.ggml.token_type'].data)
    vocab.trainer_spec.model_type = 2  # BPE
    vocab.trainer_spec.vocab_size = vocab_size
    vocab.trainer_spec.byte_fallback = True
    vocab.normalizer_spec.remove_extra_whitespaces = False
    tokens = result.fields['tokenizer.ggml.tokens']
    scores = result.fields['tokenizer.ggml.scores']
    types = result.fields['tokenizer.ggml.token_type']
    for i in range(vocab_size):
        new_token = vocab.SentencePiece()
        new_token.piece = str(bytes(tokens.parts[tokens.data[i]]),
                              encoding='utf-8')
        new_token.score = scores.parts[scores.data[i]]
        # llama.cpp tokentype is the same with sentencepiece token type
        new_token.type = int(types.parts[types.data[i]])
        vocab.pieces.append(new_token)
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
        temp_file.write(vocab.SerializeToString())
        temp_file_filename = temp_file.name
    tokenizer_args = {"vocab_file": temp_file_filename}

    if 'tokenizer.ggml.bos_token_id' in result.fields:
        tokenizer_args["bos_token"] = vocab.pieces[int(
            result.fields['tokenizer.ggml.bos_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.eos_token_id' in result.fields:
        tokenizer_args["eos_token"] = vocab.pieces[int(
            result.fields['tokenizer.ggml.eos_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.padding_token_id' in result.fields:
        tokenizer_args["pad_token"] = vocab.pieces[int(
            result.fields['tokenizer.ggml.padding_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.unknown_token_id' in result.fields:
        tokenizer_args["unk_token"] = vocab.pieces[int(
            result.fields['tokenizer.ggml.unknown_token_id'].parts[-1])].piece
    if 'tokenizer.ggml.add_bos_token' in result.fields:
        tokenizer_args["add_bos_token"] = bool(
            result.fields['tokenizer.ggml.add_bos_token'].parts[-1])
    if 'tokenizer.ggml.add_eos_token' in result.fields:
        tokenizer_args["add_eos_token"] = bool(
            result.fields['tokenizer.ggml.add_eos_token'].parts[-1])
    tokenizer = LlamaTokenizer(**tokenizer_args)
    os.unlink(temp_file_filename)
    return tokenizer


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_name.endswith("gguf"):
        return convert_gguf_to_tokenizer(tokenizer_name)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if (not trust_remote_code and
            ("does not exist or is not currently imported." in str(e)
             or "requires you to execute the tokenizer file" in str(e))):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    except AttributeError as e:
        if "BaichuanTokenizer" in str(e):
            # This is for the error "'BaichuanTokenizer' object has no
            # attribute 'sp_model'".
            tokenizer = BaichuanTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return tokenizer


def get_lora_tokenizer(lora_request: LoRARequest, *args,
                       **kwargs) -> Optional[PreTrainedTokenizer]:
    if lora_request is None:
        return None
    try:
        tokenizer = get_tokenizer(lora_request.lora_local_path, *args,
                                  **kwargs)
    except OSError as e:
        # No tokenizer was found in the LoRA folder,
        # use base model tokenizer
        logger.warning(
            f"No tokenizer found in {lora_request.lora_local_path}, "
            "using base model tokenizer instead. "
            f"(Exception: {str(e)})")
        tokenizer = None
    return tokenizer


get_lora_tokenizer_async = make_async(get_lora_tokenizer)


class TokenizerGroup:
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer_id: str, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int], **tokenizer_config):
        self.tokenizer_id = tokenizer_id
        self.tokenizer_config = tokenizer_config
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)
        if enable_lora:
            self.lora_tokenizers = LRUCache(capacity=max_num_seqs)
        else:
            self.lora_tokenizers = None

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None,
               lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = self.get_lora_tokenizer(lora_request)
        return tokenizer.encode(prompt)

    async def encode_async(
            self,
            prompt: str,
            request_id: Optional[str] = None,
            lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        return tokenizer.encode(prompt)

    def get_lora_tokenizer(
            self,
            lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            tokenizer = (get_lora_tokenizer(
                lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers.get(lora_request.lora_int_id)

    async def get_lora_tokenizer_async(
            self,
            lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            tokenizer = (await get_lora_tokenizer_async(
                lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers.get(lora_request.lora_int_id)


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    output_tokens: List[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE: The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    else:
        return "".join(sub_texts)


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int = 0,
    read_offset: int = 0,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> Tuple[List[str], str, int, int]:
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    if prev_tokens is None:
        new_tokens = tokenizer.convert_ids_to_tokens(
            all_input_ids, skip_special_tokens=skip_special_tokens)
        output_tokens = new_tokens
        # 5 is an arbitrary value that should work for all
        # tokenizers (bigger = more conservative).
        # Subtract 1 extra to account for the generated token.
        prefix_offset = max(len(output_tokens) - 6, 0)
        # If the first new token is a special token, we can't skip 1 extra token
        if skip_special_tokens and new_token_id in tokenizer.all_special_ids:
            read_offset = max(len(output_tokens), 0)
        else:
            read_offset = max(len(output_tokens) - 1, 0)
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens)
        output_tokens = prev_tokens + new_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text):]
        return new_tokens, new_text, read_offset, len(output_tokens)
    else:
        return new_tokens, "", prefix_offset, read_offset
