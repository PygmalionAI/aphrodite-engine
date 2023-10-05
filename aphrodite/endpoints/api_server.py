import argparse
import json
import os
from typing import AsyncGenerator, Dict

from fastapi import BackgroundTasks, Depends, Header, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from aphrodite.engine.args_tools import AsyncEngineArgs, EngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.

user_tokens: Dict[str, str] = {}

SECRET_TOKEN = "EMPTY"

app = FastAPI()
engine = None

def get_token(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized access.")
    token = authorization.replace("Bearer ", "")
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized access.")
    return True

@app.post("/api/v1/generate")
async def generate(request: Request, token: bool = Depends(get_token)) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
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
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

@app.get("/api/v1/model")
async def get_model_name(token: bool = Depends(get_token)) -> JSONResponse:
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