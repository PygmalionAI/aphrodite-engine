import argparse
import json
import os
from typing import AsyncGenerator, Dict

from fastapi import BackgroundTasks, Depends, Header, FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from pydantic import parse_obj_as

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.

app = FastAPI()
engine = None

valid_api_key = 'EMPTY'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/generate")
async def generate(request: Request, x_api_key: str = Header(None)) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    if x_api_key is None or x_api_key != valid_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized. Please acquire an API key.")

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    sampling_params = SamplingParams()
    
    if 'stopping_strings' in request_dict:
        request_dict['stop'] = request_dict.pop('stopping_strings')
    if 'max_new_tokens' in request_dict:
        request_dict['max_tokens'] = request_dict.pop('max_new_tokens')
    if 'ban_eos_token' in request_dict:
        request_dict['ignore_eos'] = request_dict.pop('ban_eos_token')
    if 'top_k' in request_dict and request_dict['top_k'] == 0:
        request_dict['top_k'] = -1


    for key, value in request_dict.items():
        if hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                {"text": output.text} for output in request_output.outputs
            ]
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
async def get_model_name() -> JSONResponse:
    """Return the model name based on the EngineArgs configuration."""
    if engine is not None:
        model_name = engine_args.model
        result = {"result": model_name}
        return JSONResponse(content=result)
    else:
        return JSONResponse(content={"result": "Read Only"}, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2242)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
