import os
import tempfile
from typing import Optional, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, LlamaTokenizer)
from transformers.convert_slow_tokenizer import import_protobuf
from loguru import logger

from aphrodite.lora.request import LoRARequest
from aphrodite.common.utils import make_async
from aphrodite.quantization.gguf_utils import GGUFReader
from aphrodite.transformers_utils.tokenizers import BaichuanTokenizer


def convert_gguf_to_tokenizer(checkpoint):
    if os.path.isfile(checkpoint):
        result = GGUFReader(checkpoint)
    elif os.path.isdir(checkpoint):
        try:
            return AutoTokenizer.from_pretrained(checkpoint)
        except Exception:
            pass

        all_gguf_files = sorted([
            file for file in os.listdir(checkpoint)
            if os.path.splitext(file)[-1].lower() == ".gguf"
        ])
        # assume the tokenizer is always in the first shard
        result = GGUFReader(os.path.join(checkpoint, all_gguf_files[0]))
    else:
        raise RuntimeError(f"Cannot find any tokenizer with `{checkpoint}`")

    logger.log_once("INFO", "Converting tokenizer from GGUF...")
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
    if 'tokenizer.chat_template' in result.fields:
        tokenizer_args["chat_template"] = str(
            bytes(result.fields['tokenizer.chat_template'].parts[-1]))
    tokenizer = LlamaTokenizer(**tokenizer_args)
    os.unlink(temp_file_filename)
    return tokenizer


def get_cached_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer with cached properties.
    
    This will patch the tokenizer object in place.
    
    By default, transformers will recompute multiple tokenizer
    properties each time they are called, leading to a significant
    slowdown. This function caches these properties for faster
    access."""

    tokenizer_all_special_ids = set(tokenizer.all_special_ids)
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)
    tokenizer_all_special_tokens = set(tokenizer.all_special_tokens)
    tokenizer_len = len(tokenizer)

    class CachedTokenizer(tokenizer.__class__):

        @property
        def all_special_ids(self):
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self):
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self):
            return tokenizer_all_special_tokens_extended

        def __len__(self):
            return tokenizer_len

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    tokenizer.__class__ = CachedTokenizer
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
    return get_cached_tokenizer(tokenizer)


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
