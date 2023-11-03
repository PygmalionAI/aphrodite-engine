import argparse
import json
from typing import AsyncGenerator

from fastapi import (BackgroundTasks, Header, FastAPI, HTTPException, Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.utils import random_uuid
from aphrodite.common.logits_processor import BanEOSUntil

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.

app = FastAPI()
engine = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=2242)
parser.add_argument("--api-keys", nargs="*", default=["EMPTY"])
parser.add_argument("--served-model-name", type=str, default=None)
parser = AsyncEngineArgs.add_cli_args(parser)
args = parser.parse_args()
engine_args = AsyncEngineArgs.from_cli_args(args)
valid_api_keys = args.api_keys
if args.served_model_name is not None:
    served_model = args.served_model_name
else:
    served_model = engine_args.model


@app.post("/api/v1/generate")
async def generate(
    request: Request, x_api_key: str = Header(None)) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    if x_api_key is None or x_api_key not in valid_api_keys:
        raise HTTPException(status_code=401,
                            detail="Unauthorized. Please acquire an API key.")

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    if "stopping_strings" in request_dict:
        request_dict["stop"] = request_dict.pop("stopping_strings")
    if "max_new_tokens" in request_dict:
        request_dict["max_tokens"] = request_dict.pop("max_new_tokens")
    if "min_length" in request_dict:
        request_dict["min_tokens"] = request_dict.pop("min_length")
    if "ban_eos_token" in request_dict:
        request_dict["ignore_eos"] = request_dict.pop("ban_eos_token")
    if "top_k" in request_dict and request_dict["top_k"] == 0:
        request_dict["top_k"] = -1

    request_dict["logits_processors"] = []

    min_length = request_dict.pop("min_tokens", 0)
    if request_dict.get(
            "ignore_eos",
            False):  # ignore_eos/ban_eos_token is functionally equivalent
        # to `min_tokens = max_tokens`
        min_length = request_dict.get("max_tokens", 16)

    if min_length:
        request_dict["logits_processors"].append(
            BanEOSUntil(min_length, engine.engine.tokenizer.eos_token_id))

    sampling_params = SamplingParams()
    for key, value in request_dict.items():
        if hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            # prompt = request_output.prompt
            text_outputs = [{
                "text": output.text
            } for output in request_output.outputs]
            ret = {"results": text_outputs}
            yield (json.dumps(ret) + "\n\n").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [{"text": output.text} for output in final_output.outputs]
    response_data = {"results": text_outputs}
    return JSONResponse(response_data)


@app.get("/api/v1/model")
async def get_model_name(x_api_key: str = Header(None)) -> JSONResponse:
    """Return the model name based on the EngineArgs configuration."""
    if x_api_key is None or x_api_key not in valid_api_keys:
        raise HTTPException(status_code=401,
                            detail="Unauthorized. Please acquire an API key.")
    if engine is not None:
        result = {"result": f"aphrodite/{served_model}"}
        return JSONResponse(content=result)
    else:
        return JSONResponse(content={"result": "Read Only"}, status_code=500)


@app.get("/health")
async def health() -> Response:
    """Health check route for K8s"""
    return Response(status_code=200)


if __name__ == "__main__":
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
