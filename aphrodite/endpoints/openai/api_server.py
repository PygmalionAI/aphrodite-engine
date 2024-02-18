import argparse
import asyncio
import json
from contextlib import asynccontextmanager
import os
import importlib
import inspect

from prometheus_client import make_asgi_app
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Depends, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import APIKeyHeader

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.endpoints.openai.protocol import CompletionRequest, ChatCompletionRequest, ErrorResponse
from aphrodite.common.logger import init_logger
from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
from aphrodite.endpoints.openai.serving_completions import OpenAIServingCompletion
from aphrodite.endpoints.openai.protocol import Prompt

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


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
    parser.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the header."
    )
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, Aphrodite will add it to the server using @app.middleware('http'). "
        "If a class is provided, Aphrodite will add it to the server using app.add_middleware(). "
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


async def get_api_key(api_key_header: str = Depends(api_key_header)):
    api_key = os.environ.get("APHRODITE_API_KEY") or args.api_keys
    if api_key is not None and api_key_header != "Bearer " + api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API Key")
    return api_key_header


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models(api_key: str = Depends(get_api_key)):
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.post("/v1/tokenize")
async def tokenize(prompt: Prompt, api_key: str = Depends(get_api_key)):
    tokenized = await openai_serving_chat.tokenize(prompt)
    return JSONResponse(content=tokenized)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request,
                                 api_key: str = Depends(get_api_key)):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest,
                            raw_request: Request,
                            api_key: str = Depends(get_api_key)):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)
    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(engine, served_model)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
