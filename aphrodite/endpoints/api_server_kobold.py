# Adapted from openai/api_server.py and tgi-kai-bridge

import argparse
import asyncio
import json

from http import HTTPStatus
from typing import List, Tuple, Iterator

import uvicorn
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.transformers_utils.tokenizer import get_tokenizer
from aphrodite.common.utils import random_uuid
from aphrodite.endpoints.protocol import KAIGenerationInputSchema

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model: str = "Read Only"
engine: AsyncAphrodite = None

app = FastAPI()
kai_api = APIRouter()
extra_api = APIRouter()

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"msg": message, "type": "invalid_request_error"},
                        status_code=status_code.value)

@app.exception_handler(ValueError)
def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))

def prepare_engine_payload(kai_payload: KAIGenerationInputSchema) -> Tuple[SamplingParams, List[int]]:
    """ Create SamplingParams and truncated input tokens for AsyncEngine from Kobold GenerationInput """

    if kai_payload.max_context_length > max_model_len:
        raise ValueError(
            f"max_context_length ({kai_payload.max_context_length}) must be less than or equal to "
            f"max_model_len ({max_model_len})"
        )

    sampling_params = SamplingParams(max_tokens=kai_payload.max_length)

    # KAI spec: top_k == 0 means disabled, aphrodite: top_k == -1 means disabled
    # https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings
    top_k = kai_payload.top_k if kai_payload.top_k != 0.0 else -1

    sampling_params = SamplingParams(
        n=kai_payload.n,
        best_of=kai_payload.n,
        repetition_penalty=kai_payload.rep_pen,
        temperature=kai_payload.temperature,
        tfs=kai_payload.tfs,
        top_p=kai_payload.top_p,
        top_k=top_k,
        top_a=kai_payload.top_a,
        typical_p=kai_payload.typical,
        stop=kai_payload.stop_sequence,
        ignore_eos=kai_payload.use_default_badwordsids, # TODO ban instead
        max_tokens=kai_payload.max_length,
    )

    max_input_tokens = max(1, kai_payload.max_context_length - kai_payload.max_length)
    input_tokens = tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:]

    return sampling_params, input_tokens

@kai_api.post("/generate")
async def generate(kai_payload: KAIGenerationInputSchema, raw_request: Request) -> JSONResponse:
    """ Generate text """

    req_id = f"kai-{random_uuid()}"
    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    result_generator = engine.generate(None, sampling_params, req_id, input_tokens)

    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(req_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None

    return JSONResponse({"results": list(map(lambda completion_output: {"text": completion_output.text}, final_res.outputs))})


@extra_api.post("/generate/stream")
async def generate_stream(kai_payload: KAIGenerationInputSchema, raw_request: Request) -> StreamingResponse:
    raise NotImplementedError()

def stream_kobold(iter_tokens: Iterator[str]) -> Iterator[bytes]:
    """ Produce Kobold SSE byte stream from strings """

    generated_text = ""
    for token in iter_tokens:
        if len(token) == 0:
            continue
        yield b"event: message\n"
        yield f"data: {json.dumps({'token': token})}\n\n".encode()
        generated_text += token

    return generated_text

@kai_api.get("/info/version")
async def get_version():
    """ Impersonate KAI """
    return JSONResponse({"result": "1.2.4"})

@kai_api.get("/model")
async def get_model():
    """ Get current model """
    return JSONResponse({"result": served_model})

@kai_api.get("/config/soft_prompts_list")
async def get_available_softprompts():
    """ stub for compatibility """
    return JSONResponse({"values":[]})

@kai_api.get("/config/soft_prompt")
async def get_current_softprompt():
    """ stub for compatibility """
    return JSONResponse({"value": ""})

@kai_api.put("/config/soft_prompt")
async def set_current_softprompt():
    """ stub for compatibility """
    return JSONResponse()

@extra_api.post("/abort")
async def abort_generation():
    """ stub for compatibility """
    return JSONResponse()

@extra_api.get("/version")
async def get_extra_version():
    """ Impersonate KoboldCpp with streaming support """
    return {"result": "KoboldCpp", "version": "1.30"}

app.include_router(kai_api, prefix="/api/v1")
app.include_router(kai_api, prefix="/api/latest", include_in_schema=False)
app.include_router(extra_api, prefix="/api/extra")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aphrodite KoboldAI-Compatible RESTful API server.")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.get_max_model_len()

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(engine_args.tokenizer,
                              tokenizer_mode=engine_args.tokenizer_mode,
                              trust_remote_code=engine_args.trust_remote_code)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)