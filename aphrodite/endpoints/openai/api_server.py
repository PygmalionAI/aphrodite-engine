import asyncio
import importlib
import inspect
import json
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional, Tuple

import fastapi
import uvicorn
from fastapi import APIRouter, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from loguru import logger
from prometheus_client import make_asgi_app

import aphrodite
import aphrodite.endpoints.openai.embeddings as OAIembeddings
from aphrodite.common.logger import UVICORN_LOG_CONFIG
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import _SAMPLING_EPS, SamplingParams
from aphrodite.common.utils import random_uuid
from aphrodite.endpoints.openai.args import make_arg_parser
from aphrodite.endpoints.openai.protocol import (
    ChatCompletionRequest, CompletionRequest, EmbeddingsRequest,
    EmbeddingsResponse, ErrorResponse, KAIGenerationInputSchema, Prompt)
from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
from aphrodite.endpoints.openai.serving_completions import \
    OpenAIServingCompletion
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.transformers_utils.tokenizer import get_tokenizer
from aphrodite.endpoints.openai.serving_engine import LoRA

TIMEOUT_KEEP_ALIVE = 5  # seconds

engine: Optional[AsyncAphrodite] = None
engine_args: Optional[AsyncEngineArgs] = None
openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
router = APIRouter()
kai_api = APIRouter()
extra_api = APIRouter()
kobold_lite_ui = ""
sampler_json = ""
gen_cache: dict = {}


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
router.mount("/metrics", metrics_app)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    await openai_serving_completion.engine.check_health()
    return Response(status_code=200)


@router.get("/v1/models")
async def show_available_models(x_api_key: Optional[str] = Header(None)):
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.post("/v1/tokenize")
@router.post("/v1/token/encode")
async def tokenize(request: Request,
                   prompt: Prompt,
                   x_api_key: Optional[str] = Header(None)):
    tokenized = await openai_serving_chat.tokenize(prompt)
    return JSONResponse(content=tokenized)


@router.post("/v1/detokenize")
@router.post("/v1/token/decode")
async def detokenize(request: Request,
                     token_ids: List[int],
                     x_api_key: Optional[str] = Header(None)):
    detokenized = await openai_serving_chat.detokenize(token_ids)
    return JSONResponse(content=detokenized)


@router.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def handle_embeddings(request: EmbeddingsRequest,
                            x_api_key: Optional[str] = Header(None)):
    input = request.input
    if not input:
        raise JSONResponse(
            status_code=400,
            content={"error": "Missing required argument input"})

    model = request.model if request.model else None
    response = await OAIembeddings.embeddings(input, request.encoding_format,
                                              model)
    return JSONResponse(response)


@router.get("/version", description="Fetch the Aphrodite Engine version.")
async def show_version(x_api_key: Optional[str] = Header(None)):
    ver = {"version": aphrodite.__version__}
    return JSONResponse(content=ver)


@router.get("/v1/samplers")
async def show_samplers(x_api_key: Optional[str] = Header(None)):
    """Get the available samplers."""
    global sampler_json
    if not sampler_json:
        jsonpath = os.path.dirname(os.path.abspath(__file__))
        samplerpath = os.path.join(jsonpath, "./samplers.json")
        samplerpath = os.path.normpath(samplerpath)  # Normalize the path
        if os.path.exists(samplerpath):
            with open(samplerpath, "r") as f:
                sampler_json = json.load(f)
        else:
            logger.error("Sampler JSON not found at " + samplerpath)
    return sampler_json


@router.post("/v1/lora/load")
async def load_lora(lora: LoRA, x_api_key: Optional[str] = Header(None)):
    openai_serving_chat.add_lora(lora)
    openai_serving_completion.add_lora(lora)
    if engine_args.enable_lora is False:
        logger.error("LoRA is not enabled in the engine. "
                     "Please start the server with the "
                     "--enable-lora flag.")
    return JSONResponse(content={"result": "success"})


@router.delete("/v1/lora/unload")
async def unload_lora(lora_name: str, x_api_key: Optional[str] = Header(None)):
    openai_serving_chat.remove_lora(lora_name)
    openai_serving_completion.remove_lora(lora_name)
    return JSONResponse(content={"result": "success"})


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request,
                                 x_api_key: Optional[str] = Header(None)):
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


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest,
                            raw_request: Request,
                            x_api_key: Optional[str] = Header(None)):
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


# ============ KoboldAI API ============ #


def _set_badwords(tokenizer, hf_config):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global badwordsids
    if hf_config.bad_words_ids is not None:
        badwordsids = hf_config.bad_words_ids
        return

    badwordsids = [
        v for k, v in tokenizer.get_vocab().items()
        if any(c in str(k) for c in "[]")
    ]
    if tokenizer.pad_token_id in badwordsids:
        badwordsids.remove(tokenizer.pad_token_id)
    badwordsids.append(tokenizer.eos_token_id)


def prepare_engine_payload(
        kai_payload: KAIGenerationInputSchema
) -> Tuple[SamplingParams, List[int]]:
    """Create SamplingParams and truncated input tokens for AsyncEngine"""

    if not kai_payload.genkey:
        kai_payload.genkey = f"kai-{random_uuid()}"

    # if kai_payload.max_context_length > engine_args.max_model_len:
    #     raise ValueError(
    #         f"max_context_length ({kai_payload.max_context_length}) "
    #         "must be less than or equal to "
    #         f"max_model_len ({engine_args.max_model_len})")

    kai_payload.top_k = kai_payload.top_k if kai_payload.top_k != 0.0 else -1
    kai_payload.tfs = max(_SAMPLING_EPS, kai_payload.tfs)
    if kai_payload.temperature < _SAMPLING_EPS:
        kai_payload.n = 1
        kai_payload.top_p = 1.0
        kai_payload.top_k = -1
    if kai_payload.dynatemp_range is not None:
        dynatemp_min = kai_payload.temperature - kai_payload.dynatemp_range
        dynatemp_max = kai_payload.temperature + kai_payload.dynatemp_range

    sampling_params = SamplingParams(
        n=kai_payload.n,
        best_of=kai_payload.n,
        repetition_penalty=kai_payload.rep_pen,
        temperature=kai_payload.temperature,
        dynatemp_min=dynatemp_min if kai_payload.dynatemp_range > 0 else 0.0,
        dynatemp_max=dynatemp_max if kai_payload.dynatemp_range > 0 else 0.0,
        dynatemp_exponent=kai_payload.dynatemp_exponent,
        smoothing_factor=kai_payload.smoothing_factor,
        smoothing_curve=kai_payload.smoothing_curve,
        tfs=kai_payload.tfs,
        top_p=kai_payload.top_p,
        top_k=kai_payload.top_k,
        top_a=kai_payload.top_a,
        min_p=kai_payload.min_p,
        typical_p=kai_payload.typical,
        eta_cutoff=kai_payload.eta_cutoff,
        epsilon_cutoff=kai_payload.eps_cutoff,
        mirostat_mode=kai_payload.mirostat,
        mirostat_tau=kai_payload.mirostat_tau,
        mirostat_eta=kai_payload.mirostat_eta,
        stop=kai_payload.stop_sequence,
        include_stop_str_in_output=kai_payload.include_stop_str_in_output,
        custom_token_bans=badwordsids
        if kai_payload.use_default_badwordsids else [],
        max_tokens=kai_payload.max_length,
        seed=kai_payload.sampler_seed,
    )

    max_input_tokens = max(
        1, kai_payload.max_context_length - kai_payload.max_length)
    input_tokens = tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:]

    return sampling_params, input_tokens


@kai_api.post("/generate")
async def generate(kai_payload: KAIGenerationInputSchema) -> JSONResponse:
    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    result_generator = engine.generate(None, sampling_params,
                                       kai_payload.genkey, input_tokens)

    final_res: RequestOutput = None
    previous_output = ""
    async for res in result_generator:
        final_res = res
        new_chunk = res.outputs[0].text[len(previous_output):]
        previous_output += new_chunk
        gen_cache[kai_payload.genkey] = previous_output

    assert final_res is not None
    del gen_cache[kai_payload.genkey]

    return JSONResponse(
        {"results": [{
            "text": output.text
        } for output in final_res.outputs]})


@extra_api.post("/generate/stream")
async def generate_stream(
        kai_payload: KAIGenerationInputSchema) -> StreamingResponse:

    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    results_generator = engine.generate(None, sampling_params,
                                        kai_payload.genkey, input_tokens)

    async def stream_kobold() -> AsyncGenerator[bytes, None]:
        previous_output = ""
        async for res in results_generator:
            new_chunk = res.outputs[0].text[len(previous_output):]
            previous_output += new_chunk
            yield b"event: message\n"
            yield f"data: {json.dumps({'token': new_chunk})}\n\n".encode()

    return StreamingResponse(stream_kobold(),
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive",
                             },
                             media_type="text/event-stream")


@extra_api.post("/generate/check")
@extra_api.get("/generate/check")
async def check_generation(request: Request):
    text = ""
    try:
        request_dict = await request.json()
        if "genkey" in request_dict and request_dict["genkey"] in gen_cache:
            text = gen_cache[request_dict["genkey"]]
    except json.JSONDecodeError:
        pass

    return JSONResponse({"results": [{"text": text}]})


@extra_api.post("/abort")
async def abort_generation(request: Request):
    try:
        request_dict = await request.json()
        if "genkey" in request_dict:
            await engine.abort(request_dict["genkey"])
    except json.JSONDecodeError:
        pass

    return JSONResponse({})


@extra_api.post("/tokencount")
async def count_tokens(request: Request):
    """Tokenize string and return token count"""

    request_dict = await request.json()
    tokenizer_result = await openai_serving_chat.tokenize(
        Prompt(**request_dict))
    return JSONResponse({"value": tokenizer_result["value"]})


@kai_api.get("/info/version")
async def get_version():
    """Impersonate KAI"""
    return JSONResponse({"result": "1.2.4"})


@kai_api.get("/model")
async def get_model():
    return JSONResponse({"result": f"aphrodite/{served_model}"})


@kai_api.get("/config/soft_prompts_list")
async def get_available_softprompts():
    """Stub for compatibility"""
    return JSONResponse({"values": []})


@kai_api.get("/config/soft_prompt")
async def get_current_softprompt():
    """Stub for compatibility"""
    return JSONResponse({"value": ""})


@kai_api.put("/config/soft_prompt")
async def set_current_softprompt():
    """Stub for compatibility"""
    return JSONResponse({})


@kai_api.get("/config/max_length")
async def get_max_length() -> JSONResponse:
    max_length = args.max_length
    return JSONResponse({"value": max_length})


@kai_api.get("/config/max_context_length")
@extra_api.get("/true_max_context_length")
async def get_max_context_length() -> JSONResponse:
    max_context_length = engine_args.max_model_len
    return JSONResponse({"value": max_context_length})


@extra_api.get("/preloadstory")
async def get_preloaded_story() -> JSONResponse:
    """Stub for compatibility"""
    return JSONResponse({})


@extra_api.get("/version")
async def get_extra_version():
    """Impersonate KoboldCpp"""
    return JSONResponse({"result": "KoboldCpp", "version": "1.63"})


@router.get("/")
async def get_kobold_lite_ui():
    """Serves a cached copy of the Kobold Lite UI, loading it from disk
    on demand if needed."""
    global kobold_lite_ui
    if kobold_lite_ui == "":
        scriptpath = os.path.dirname(os.path.abspath(__file__))
        klitepath = os.path.join(scriptpath, "../kobold/klite.embd")
        klitepath = os.path.normpath(klitepath)  # Normalize the path
        if os.path.exists(klitepath):
            with open(klitepath, "r") as f:
                kobold_lite_ui = f.read()
        else:
            logger.error("Kobold Lite UI not found at " + klitepath)
    return HTMLResponse(content=kobold_lite_ui)


# ============ KoboldAI API ============ #


def build_app(args):
    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path
    if args.launch_kobold_api:
        logger.warning("Launching Kobold API server in addition to OpenAI. "
                       "Keep in mind that the Kobold API routes are NOT "
                       "protected via the API key.")
        app.include_router(kai_api, prefix="/api/v1")
        app.include_router(kai_api,
                           prefix="/api/latest",
                           include_in_schema=False)
        app.include_router(extra_api, prefix="/api/extra")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = openai_serving_completion.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := os.environ.get("APHRODITE_API_KEY") or args.api_keys:
        admin_key = os.environ.get("APHRODITE_ADMIN_KEY") or args.admin_key

        if admin_key is None:
            logger.warning("Admin key not provided. Admin operations will "
                           "be disabled.")

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            excluded_paths = ["/api"]
            if any(
                    request.url.path.startswith(path)
                    for path in excluded_paths):
                return await call_next(request)
            if not request.url.path.startswith("/v1"):
                return await call_next(request)

            # Browsers may send OPTIONS requests to check CORS headers
            # before sending the actual request. We should allow these
            # requests to pass through without authentication.
            # See https://github.com/PygmalionAI/aphrodite-engine/issues/434
            if request.method == "OPTIONS":
                return await call_next(request)

            auth_header = request.headers.get("Authorization")
            api_key_header = request.headers.get("x-api-key")

            if request.url.path.startswith("/v1/lora"):
                if admin_key is not None and api_key_header == admin_key:
                    return await call_next(request)
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)

            if auth_header != "Bearer " + token and api_key_header != token:
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
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


def run_server(args):
    app = build_app(args)

    logger.debug(f"args: {args}")

    global engine, engine_args, openai_serving_chat, openai_serving_completion,\
        tokenizer, served_model
    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=engine_args.trust_remote_code,
        revision=engine_args.revision,
    )

    chat_template = args.chat_template
    if chat_template is None and tokenizer.chat_template is not None:
        chat_template = tokenizer.chat_template

    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model, args.lora_modules)
    engine_model_config = asyncio.run(engine.get_model_config())

    if args.launch_kobold_api:
        _set_badwords(tokenizer, engine_model_config.hf_config)
    try:
        uvicorn.run(app,
                    host=args.host,
                    port=args.port,
                    log_level="info",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile,
                    log_config=UVICORN_LOG_CONFIG)
    except KeyboardInterrupt:
        logger.info("API server stopped by user. Exiting.")
    except asyncio.exceptions.CancelledError:
        logger.info("API server stopped due to a cancelled request. Exiting.")


if __name__ == "__main__":
    # NOTE:
    # This section should be in sync with aphrodite/endpoints/cli.py
    # for CLI entrypoints.
    parser = make_arg_parser()
    args = parser.parse_args()
    run_server(args)
