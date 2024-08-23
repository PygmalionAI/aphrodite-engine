import codecs
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Awaitable, Iterable, List, Optional, TypedDict, cast, final

import requests
from loguru import logger
from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartTextParam)

from aphrodite.endpoints.openai.protocol import (
    ChatCompletionContentPartParam, ChatCompletionMessageParam)
from aphrodite.endpoints.openai.serving_engine import OpenAIServing
from aphrodite.multimodal import MultiModalDataDict
from aphrodite.multimodal.utils import async_get_and_parse_image


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    mm_futures: List[Awaitable[MultiModalDataDict]] = field(
        default_factory=list)


def load_chat_template(self, chat_template: Optional[str]):
    tokenizer = self.tokenizer

    if chat_template is not None:
        try:
            if chat_template.startswith('http'):
                response = requests.get(chat_template)
                if response.status_code == 200:
                    tokenizer.chat_template = response.text
                else:
                    raise ValueError("Failed to download chat template "
                                     f"from {chat_template}")
            else:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
        except OSError as e:
            JINJA_CHARS = "{}\n"
            if not any(c in chat_template for c in JINJA_CHARS):
                msg = (f"The supplied chat template ({chat_template}) "
                       f"looks like a file path, but it failed to be "
                       f"opened. Reason: {e}")
                raise ValueError(msg) from e

            # If opening a file fails, set chat template to be args to
            # ensure we decode so our escape are interpreted correctly
            tokenizer.chat_template = codecs.decode(chat_template,
                                                    "unicode_escape")

        logger.info("Using supplied chat template")
    elif tokenizer.chat_template is not None:
        logger.info("Using default chat template")
    else:
        logger.warning("No chat template provided. Chat API will not work.")


@lru_cache(maxsize=None)
def _image_token_str(engine: OpenAIServing) -> Optional[str]:
    # TODO: Let user specify how to insert image tokens into prompt
    # (similar to chat template)
    model_type = engine.model_config.hf_config.model_type
    if model_type == "phi3_v":
        # Workaround since this token is not defined in the tokenizer
        return "<|image_1|>"
    if model_type in ("blip-2", "chatglm", "fuyu", "minicpmv", "paligemma"):
        # These models do not use image tokens in the prompt
        return None
    if model_type.startswith("llava"):
        return engine.tokenizer.decode(
            engine.model_config.hf_config.image_token_index)

    else:
        raise TypeError(f"Unknown model type: {model_type}")


# TODO: Let user specify how to insert image tokens into prompt
# (similar to chat template)
def _get_full_image_text_prompt(engine: OpenAIServing, image_token_str: str,
                                text_prompt: str) -> str:
    """Combine image and text prompts for vision language model"""

    # NOTE: For now we assume all model architectures use the same
    # image + text prompt format. This may change in the future.
    return f"{image_token_str}\n{text_prompt}"


def _parse_chat_message_content_parts(
    engine: OpenAIServing,
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
) -> ChatMessageParseResult:
    texts: List[str] = []
    mm_futures: List[Awaitable[MultiModalDataDict]] = []

    for part in parts:
        part_type = part["type"]
        if part_type == "text":
            text = cast(ChatCompletionContentPartTextParam, part)["text"]
            texts.append(text)
        elif part_type == "image_url":
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple 'image_url' input is currently not supported.")

            image_url = cast(ChatCompletionContentPartImageParam,
                             part)["image_url"]

            if image_url.get("detail", "auto") != "auto":
                logger.warning(
                    "'image_url.detail' is currently not supported and "
                    "will be ignored.")

            image_future = async_get_and_parse_image(image_url["url"])
            mm_futures.append(image_future)
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)

    if mm_futures:
        image_token_str = _image_token_str(engine)
        if image_token_str is not None:
            if image_token_str in text_prompt:
                logger.warning(
                    "Detected image token string in the text prompt. "
                    "Skipping prompt formatting.")
            else:
                text_prompt = _get_full_image_text_prompt(
                    engine,
                    image_token_str=image_token_str,
                    text_prompt=text_prompt,
                )

    messages = [ConversationMessage(role=role, content=text_prompt)]

    return ChatMessageParseResult(messages=messages, mm_futures=mm_futures)


def parse_chat_message_content(
    engine: OpenAIServing,
    message: ChatCompletionMessageParam,
) -> ChatMessageParseResult:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return ChatMessageParseResult(messages=[], mm_futures=[])
    if isinstance(content, str):
        messages = [ConversationMessage(role=role, content=content)]
        return ChatMessageParseResult(messages=messages, mm_futures=[])

    return _parse_chat_message_content_parts(engine, role, content)
