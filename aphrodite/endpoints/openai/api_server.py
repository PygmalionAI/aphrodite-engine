import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
import os
import importlib
import inspect

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.engine.metrics import add_global_metrics_labels
from aphrodite.endpoints.openai.protocol import (
    CompletionRequest, ChatCompletionRequest, ErrorResponse, Prompt)
from aphrodite.common.logger import init_logger
from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
from aphrodite.endpoints.openai.serving_completions import OpenAIServingCompletion
from aphrodite.endpoints.openai.tools import OpenAIToolsPrompter

TIMEOUT_KEEP_ALIVE = 5  # seconds

aphrodite_engine = None
aphrodite_engine_args = None
openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await aphrodite_engine.do_log_stats()

    if not aphrodite_engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aphrodite OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
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
    parser.add_argument("--tools-template",
                        type=str,
                        default=None,
                        help="The file path to alternative tools template")
    parser.add_argument("--enable-api-tools",
                        action="store_true",
                        help="Enable OpenAI-like tools API "
                        "(only function calls are currently supported)")
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
        "--dev-mode",
        action="store_true",
        help=
        "Enable API internals and templates reloading but do not deallocate the engine. This should only be used for development purpose."
    )
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

def _loadServingServices():
    """ Load or reload the OpenAI service.
        This function should only be called once on initialization, but may be called to reload the API internals.
        Reloading must be used for development purpose only. """
    global openai_serving_chat
    global openai_serving_completion
    if openai_serving_chat is not None:
        del openai_serving_chat
    if openai_serving_completion is not None:
        del openai_serving_completion

    openai_tools_prompter = OpenAIToolsPrompter(
        template_path=args.tools_template) if args.enable_api_tools else None
    openai_serving_chat = OpenAIServingChat(
        engine=aphrodite_engine,
        served_model=served_model,
        response_role=args.response_role,
        chat_template=args.chat_template,
        openai_tools_prompter=openai_tools_prompter,
        dev_mode=args.dev_mode)
    openai_serving_completion = OpenAIServingCompletion(
        aphrodite_engine, served_model)

app.add_middleware(MetricsMiddleware)  # Trace HTTP server metrics
app.add_route("/metrics", metrics)  # Exposes HTTP metrics


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

if "--dev-mode" in sys.argv:

    @app.get("/privileged")
    async def privileged() -> Response:
        """Reload the API internals. Dangerous!"""
        logger.warning("privileged called.")
        _loadServingServices()
        return Response(status_code=200)

@app.post("/v1/tokenize")
async def tokenize(prompt: Prompt):
    tokenized = await openai_serving_chat.tokenize_text(prompt)
    return JSONResponse(content=tokenized.model_dump())

@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
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
async def create_completion(request: CompletionRequest, raw_request: Request):
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

    if token := os.environ.get("APHRODITE_API_KEY") or args.api_keys:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

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
    if args.dev_mode:
        logger.warning(
            "\n"
            "######################################################################\n"
            "dev-mode enabled. This should only be used for development purpose.\n"
            "If It's not the case, you should disable this!\n"
            "######################################################################\n"
        )

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    aphrodite_engine_args = AsyncEngineArgs.from_cli_args(args)
    aphrodite_engine = AsyncAphrodite.from_engine_args(aphrodite_engine_args)
    _loadServingServices()

    # Register labels for metrics
    add_global_metrics_labels(model_name=aphrodite_engine_args.model)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
