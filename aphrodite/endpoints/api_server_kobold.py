# Adapted from openai/api_server.py and tgi-kai-bridge

import argparse
import asyncio
import json
import os

from http import HTTPStatus
from typing import List, Tuple, AsyncGenerator

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams, _SAMPLING_EPS
from aphrodite.transformers_utils.tokenizer import get_tokenizer
from aphrodite.common.utils import random_uuid
from aphrodite.endpoints.protocol import KAIGenerationInputSchema

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model: str = "Read Only"
engine: AsyncAphrodite = None

app = FastAPI()
kai_api = APIRouter()
extra_api = APIRouter()
kobold_lite_ui = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"msg": message, "type": "invalid_request_error"},
                        status_code=status_code.value)

@app.exception_handler(ValueError)
def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))

def prepare_engine_payload(kai_payload: KAIGenerationInputSchema) -> Tuple[SamplingParams, List[int]]:
    """Create SamplingParams and truncated input tokens for AsyncEngine"""

    if kai_payload.max_context_length > max_model_len:
        raise ValueError(
            f"max_context_length ({kai_payload.max_context_length}) must be less than or equal to "
            f"max_model_len ({max_model_len})"
        )

    sampling_params = SamplingParams(max_tokens=kai_payload.max_length)

    # KAI spec: top_k == 0 means disabled, aphrodite: top_k == -1 means disabled
    # https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings
    kai_payload.top_k = kai_payload.top_k if kai_payload.top_k != 0.0 else -1
    kai_payload.tfs = max(_SAMPLING_EPS, kai_payload.tfs)
    if kai_payload.temperature < _SAMPLING_EPS:
        # temp < _SAMPLING_EPS: greedy sampling
        kai_payload.n = 1
        kai_payload.top_p = 1.0
        kai_payload.top_k = -1


    sampling_params = SamplingParams(
        n=kai_payload.n,
        best_of=kai_payload.n,
        repetition_penalty=kai_payload.rep_pen,
        temperature=kai_payload.temperature,
        tfs=kai_payload.tfs,
        top_p=kai_payload.top_p,
        top_k=kai_payload.top_k,
        top_a=kai_payload.top_a,
        typical_p=kai_payload.typical,
        eta_cutoff=kai_payload.eta_cutoff,
        epsilon_cutoff=kai_payload.eps_cutoff,
        stop=kai_payload.stop_sequence,
        # ignore_eos=kai_payload.use_default_badwordsids, # TODO ban instead
        max_tokens=kai_payload.max_length,
    )

    max_input_tokens = max(1, kai_payload.max_context_length - kai_payload.max_length)
    input_tokens = tokenizer(kai_payload.prompt).input_ids[-max_input_tokens:]

    return sampling_params, input_tokens

@kai_api.post("/generate")
async def generate(kai_payload: KAIGenerationInputSchema) -> JSONResponse:
    """ Generate text """

    req_id = f"kai-{random_uuid()}"
    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    result_generator = engine.generate(None, sampling_params, req_id, input_tokens)

    final_res: RequestOutput = None
    async for res in result_generator:
        final_res = res
    assert final_res is not None

    return JSONResponse({"results": [{"text": output.text} for output in final_res.outputs]})


@extra_api.post("/generate/stream")
async def generate_stream(kai_payload: KAIGenerationInputSchema) -> StreamingResponse:
    """ Generate text SSE streaming """

    req_id = f"kai-{random_uuid()}"
    sampling_params, input_tokens = prepare_engine_payload(kai_payload)
    results_generator = engine.generate(None, sampling_params, req_id, input_tokens)

    async def stream_kobold() -> AsyncGenerator[bytes, None]:
        previous_output = ""
        async for res in results_generator:
            new_chunk = res.outputs[0].text[len(previous_output):]
            previous_output += new_chunk
            yield b"event: message\n"
            yield f"data: {json.dumps({'token': new_chunk})}\n\n".encode()

    return StreamingResponse(stream_kobold(),
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                             media_type='text/event-stream')

@extra_api.post("/generate/check")
async def check_generation():
    """ stub for compatibility """
    return JSONResponse({"results": [{"text": ""}]})

@kai_api.get("/info/version")
async def get_version():
    """ Impersonate KAI """
    return JSONResponse({"result": "1.2.4"})

@kai_api.get("/model")
async def get_model():
    """ Get current model """
    return JSONResponse({"result": f"aphrodite/{served_model}"})

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
    return JSONResponse({})

@app.get("/api/latest/config/max_context_length")
async def get_max_context_length() -> JSONResponse:
    """Return the max context length based on the EngineArgs configuration."""
    max_context_length = engine_model_config.max_model_len
    return JSONResponse({"value": max_context_length })

@app.get("/api/latest/config/max_length")
async def get_max_length() -> JSONResponse:
    """Why do we need this twice?"""
    max_length = args.max_length
    return JSONResponse({"value": max_length})

@extra_api.post("/abort")
async def abort_generation():
    """ stub for compatibility """
    return JSONResponse({})

@extra_api.get("/version")
async def get_extra_version():
    """ Impersonate KoboldCpp with streaming support """
    return JSONResponse({"result": "KoboldCpp", "version": "1.30"})

@app.get("/")
async def get_kobold_lite_ui():
    """Serves a cached copy of the Kobold Lite UI, loading it from disk on demand if needed."""
    #read and return embedded kobold lite
    global kobold_lite_ui
    if kobold_lite_ui=="":
        scriptpath = os.path.dirname(os.path.abspath(__file__))
        klitepath = os.path.join(scriptpath, "klite.embd")
        if os.path.exists(klitepath):
            with open(klitepath, "r") as f:
                kobold_lite_ui = f.read()
        else:
            print("Embedded Kobold Lite not found")
    return HTMLResponse(content=kobold_lite_ui)

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
    parser.add_argument("--port", type=int, default=2242, help="port number")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--max-length",
                    type=int,
                    default=256,
                    help="The maximum length of the generated text. "
                    "For use with Kobold Horde.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    global args
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