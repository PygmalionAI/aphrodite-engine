# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import json
from contextlib import asynccontextmanager

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request, Response, Header, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.engine.metrics import add_global_metrics_labels
from aphrodite.endpoints.openai.protocol import (
    CompletionRequest, ChatCompletionRequest, ErrorResponse)
from aphrodite.common.logger import init_logger
from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
from aphrodite.endpoints.openai.serving_completion import OpenAIServingCompletion
from aphrodite.transformers_utils.tokenizer import get_tokenizer

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None

logger = init_logger(__name__)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()
    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())
    yield

app = fastapi.FastAPI(lifespan=lifespan)

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
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="SSL key file path.")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="SSL cert file path.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


app.add_middleware(MetricsMiddleware)  # trace HTTP server metrics
app.add_route("/metrics", metrics)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.dict(), status_code=HTTPStatus.BAD_REQUEST)

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


@app.get("/health")
async def health() -> Response:
    """Health check route for K8s"""
    return Response(status_code=200)

class Prompt(BaseModel):
    prompt: str


@app.post("/v1/tokenize")
async def tokenize_text(
    prompt: Prompt,
    # pylint: disable=unused-argument
    api_key: str = Depends(_verify_api_key)):
    """Tokenize prompt using the tokenizer.
    Returns:
        value: The number of tokens in the prompt.
        ids: The token IDs of the prompt.
    """
    try:
        tokenized_prompt = tokenizer.tokenize(prompt.prompt)
        token_ids = tokenizer.convert_tokens_to_ids(tokenized_prompt)
        return {"value": len(tokenized_prompt), "ids": token_ids}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/v1/models")
async def show_available_models(
    # pylint: disable=unused-argument
    api_key: str = Depends(_verify_api_key)
):
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.dict())


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    # pylint: disable=unused-argument
    api_key: str = Depends(_verify_api_key)):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if request.stream and not isinstance(generator, ErrorResponse):
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.dict())


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    raw_request: Request,
    # pylint: disable=unused-argument
    api_key: str = Depends(_verify_api_key)):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    logger.info("TYPE COMPLETION: %s" %str(type(generator)))
    if request.stream and not isinstance(generator, ErrorResponse):
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.dict())


if __name__ == "__main__":
    args = parse_args()
    global EXPECTED_API_KEYS
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


    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)
    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(engine, served_model,
                                                        args.response_role,
                                                        args.chat_template)
     # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=engine_args.trust_remote_code)

    add_global_metrics_labels(model_name=engine_args.model)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
