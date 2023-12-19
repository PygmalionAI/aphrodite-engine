# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import codecs
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from fastapi import Request, Response, Header, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.engine.metrics import add_global_metrics_labels
from aphrodite.endpoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.transformers_utils.tokenizer import get_tokenizer
from aphrodite.common.utils import random_uuid
from aphrodite.common.logits_processor import BiasLogitsProcessor

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()
engine = None
response_role = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aphrodite OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="host name")
    parser.add_argument("--port", type=int, default=2242, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--api-keys",
                        nargs="*",
                        help="Authorization API Keys for the server.")
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model.")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=True.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


app.add_middleware(MetricsMiddleware)  # trace HTTP server metrics
app.add_route("/metrics", metrics)


def _verify_api_key(x_api_key: str = Header(None),
                    authorization: str = Header(None)):
    if not EXPECTED_API_KEYS:  # If no keys are provided
        return "NoKey"  # Return a default value
    if x_api_key and x_api_key in EXPECTED_API_KEYS:
        return x_api_key
    elif authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token in EXPECTED_API_KEYS:
            return token
    raise HTTPException(
        status_code=401,
        detail="Invalid API Key",
    )


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


def load_chat_template(args, tokenizer):  # pylint: disable=redefined-outer-name
    if args.chat_template is not None:
        try:
            with open(args.chat_template, "r") as f:
                chat_template = f.read()
        except OSError:
            # If opening a file fails, set chat template to be args to
            # ensure we decode so our escape are interpreted correctly
            chat_template = codecs.decode(args.chat_template, "unicode_escape")

        tokenizer.chat_template = chat_template
        logger.info(
            f"Using supplied chat template:\n{tokenizer.chat_template}")
    elif tokenizer.chat_template is not None:
        logger.info(f"Using default chat template:\n{tokenizer.chat_template}")
    else:
        logger.warning("No chat template provided. Chat API will not work.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    input_ids = prompt_ids if prompt_ids is not None else tokenizer(
        prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check route for K8s"""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models(
        # pylint: disable=unused-argument
        api_key: str = Depends(_verify_api_key)):
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(
    token_ids: List[int],
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None,
    num_output_top_logprobs: Optional[int] = None,
    initial_text_offset: int = 0,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    if num_output_top_logprobs:
        logprobs.top_logprobs = []
    for i, token_id in enumerate(token_ids):
        step_top_logprobs = top_logprobs[i]
        if step_top_logprobs is not None:
            token_logprob = step_top_logprobs[token_id]
        else:
            token_logprob = None
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(token_logprob)
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        if num_output_top_logprobs:
            logprobs.top_logprobs.append({
                tokenizer.convert_ids_to_tokens(i): p
                for i, p in step_top_logprobs.items()
            } if step_top_logprobs else None)
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    # pylint: disable=unused-argument
    api_key: str = Depends(_verify_api_key)):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
    """

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    try:
        prompt = tokenizer.apply_chat_template(
            conversation=request.messages,
            tokenize=False,
            add_generation_prompt=request.add_generation_prompt)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Error in applying chat template from request: {str(e)}")
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    if not request.logit_bias:
        logit_processors = []
    else:
        biases = dict(
            map(lambda bias: (int(bias[0]), bias[1]),
                request.logit_bias.items()))
        logit_processors = [BiasLogitsProcessor(biases)]

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    chunk_object_type = "chat.completion.chunk"

    # We disable top_k at -1, add this conversion for
    # compatibility
    if request.top_k == 0:
        request.top_k = -1
    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            top_a=request.top_a,
            min_p=request.min_p,
            tfs=request.tfs,
            eta_cutoff=request.eta_cutoff,
            epsilon_cutoff=request.epsilon_cutoff,
            typical_p=request.typical_p,
            mirostat_mode=request.mirostat_mode,
            mirostat_tau=request.mirostat_tau,
            mirostat_eta=request.mirostat_eta,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            include_stop_str_in_output=request.include_stop_str_in_output,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=request.
            spaces_between_special_tokens,  # pylint: disable=line-too-long
            custom_token_bans=request.custom_token_bans,
            logprobs=request.logprobs,
            prompt_logprobs=request.prompt_logprobs,
            logits_processors=logit_processors,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params, request_id,
                                       token_ids)

    def get_role() -> str:
        if request.add_generation_prompt:
            return response_role
        else:
            return request.messages[-1]["role"]

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # Send first response for each request.n (index) with the role
        role = get_role()
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None)
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 object=chunk_object_type,
                                                 created=created_time,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index

                if finish_reason_sent[i]:
                    continue

                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                else:
                    # Send the finish response for each request.n only once
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=previous_num_tokens[i],
                        total_tokens=prompt_tokens + previous_num_tokens[i],
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i, delta=[], finish_reason=output.finish_reason)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    if final_usage is not None:
                        chunk.usage = final_usage
                    data = chunk.json(exclude_unset=True,
                                      exclude_none=True,
                                      ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def completion_full_generator():
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             "Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = get_role()
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")
    else:
        return await completion_full_generator()


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    raw_request: Request,
    # pylint: disable=unused-argument
    api_key: str = Depends(_verify_api_key)):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the Aphrodite engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
    """

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if not request.logit_bias:
        logit_processors = []
    else:
        biases = dict(
            map(lambda bias: (int(bias[0]), bias[1]),
                request.logit_bias.items()))
        logit_processors = [BiasLogitsProcessor(biases)]

    # OpenAI API supports echoing the prompt when max_tokens is 0.
    echo_without_generation = request.echo and request.max_tokens == 0

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        _, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())

    # We disable top_k at -1, add this conversion for
    # compatibility
    if request.top_k == 0:
        request.top_k = -1

    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            top_a=request.top_a,
            min_p=request.min_p,
            tfs=request.tfs,
            eta_cutoff=request.eta_cutoff,
            epsilon_cutoff=request.epsilon_cutoff,
            typical_p=request.typical_p,
            mirostat_mode=request.mirostat_mode,
            mirostat_tau=request.mirostat_tau,
            mirostat_eta=request.mirostat_eta,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            include_stop_str_in_output=request.include_stop_str_in_output,
            max_tokens=request.max_tokens
            if not echo_without_generation else 1,
            best_of=request.best_of,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=request.
            spaces_between_special_tokens,  # pylint: disable=line-too-long
            custom_token_bans=request.custom_token_bans,
            logprobs=request.logprobs,
            prompt_logprobs=request.prompt_logprobs if request.echo else None,
            logits_processors=logit_processors,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = engine.generate(None,
                                           sampling_params,
                                           request_id,
                                           prompt_token_ids=prompt)
    else:
        result_generator = engine.generate(prompt, sampling_params, request_id,
                                           token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        response_json = response.json(exclude_unset=True, ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        has_echoed = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                token_ids = output.token_ids[previous_num_tokens[i]:]
                if request.logprobs is not None:
                    top_logprobs = output.logprobs[previous_num_tokens[i]:]
                else:
                    top_logprobs = None
                offsets = len(previous_texts[i])
                if request.echo and not has_echoed[i]:
                    if not echo_without_generation:
                        delta_text = res.prompt + delta_text
                        token_ids = res.prompt_token_ids + token_ids
                        if top_logprobs:
                            top_logprobs = res.prompt_logprobs + top_logprobs
                    else:  # only just return the prompt
                        delta_text = res.prompt
                        token_ids = res.prompt_token_ids
                        if top_logprobs:
                            top_logprobs = res.prompt_logprobs
                    has_echoed[i] = True
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        token_ids=token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                        initial_text_offset=offsets,
                    )
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                finish_reason = output.finish_reason
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    prompt_tokens = len(res.prompt_token_ids)
                    completion_tokens = len(output.token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    prompt_token_ids = final_res.prompt_token_ids
    prompt_logprobs = final_res.prompt_logprobs
    prompt_text = final_res.prompt
    for output in final_res.outputs:
        if request.logprobs is not None:
            if not echo_without_generation:
                token_ids = output.token_ids
                top_logprobs = output.logprobs
                if request.echo:
                    token_ids = prompt_token_ids + token_ids
                    top_logprobs = prompt_logprobs + top_logprobs
            else:
                token_ids = prompt_token_ids
                top_logprobs = prompt_logprobs
            logprobs = create_logprobs(
                token_ids=token_ids,
                top_logprobs=top_logprobs,
                num_output_top_logprobs=request.logprobs,
            )
        else:
            logprobs = None
        if not echo_without_generation:
            output_text = output.text
            if request.echo:
                output_text = prompt_text + output_text
        else:
            output_text = prompt_text
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output_text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    args = parse_args()
    global EXPECTED_API_KEYS  # pylint: disable=global-at-module-level
    EXPECTED_API_KEYS = args.api_keys
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.debug(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    response_role = args.response_role

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code)

    load_chat_template(args, tokenizer)

    add_global_metrics_labels(model_name=engine_args.model)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
