import asyncio
from io import StringIO
from typing import Awaitable, List

import aiohttp
from loguru import logger

from aphrodite.common.utils import FlexibleArgumentParser, random_uuid
from aphrodite.endpoints.openai.protocol import (BatchRequestInput,
                                                 BatchRequestOutput,
                                                 BatchResponseData,
                                                 ChatCompletionResponse,
                                                 ErrorResponse)
from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.version import __version__ as APHRODITE_VERSION


def parse_args():
    parser = FlexibleArgumentParser(
        description="Aphrodite OpenAI-Compatible batch runner.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help=
        "The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.")
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help="The path or url to a single output file. Currently supports "
        "local file paths, or web (http or https) urls. If a URL is specified,"
        " the file should be available via HTTP PUT.")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return f.read()


async def write_file(path_or_url: str, data: str) -> None:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.put(path_or_url, data=data.encode("utf-8")):
            pass
    else:
        # We should make this async, but as long as this is always run as a
        # standalone program, blocking the event loop won't effect performance
        # in this particular case.
        with open(path_or_url, "w", encoding="utf-8") as f:
            f.write(data)


async def run_request(chat_serving: OpenAIServingChat,
                      request: BatchRequestInput) -> BatchRequestOutput:
    chat_request = request.body
    chat_response = await chat_serving.create_chat_completion(chat_request)

    if isinstance(chat_response, ChatCompletionResponse):
        batch_output = BatchRequestOutput(
            id=f"aphrodite-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                body=chat_response,
                request_id=f"aphrodite-batch-{random_uuid()}"),
            error=None,
        )
    elif isinstance(chat_response, ErrorResponse):
        batch_output = BatchRequestOutput(
            id=f"aphrodite-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                status_code=chat_response.code,
                request_id=f"aphrodite-batch-{random_uuid()}"),
            error=chat_response,
        )
    else:
        raise ValueError("Request must not be sent in stream mode")

    return batch_output


async def main(args):
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncAphrodite.from_engine_args(engine_args)

    # When using single Aphrodite without engine_use_ray
    model_config = await engine.get_model_config()

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
    )

    # Submit all requests in the file to the engine "concurrently".
    response_futures: List[Awaitable[BatchRequestOutput]] = []
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        request = BatchRequestInput.model_validate_json(request_json)
        response_futures.append(run_request(openai_serving_chat, request))

    responses = await asyncio.gather(*response_futures)

    output_buffer = StringIO()
    for response in responses:
        print(response.model_dump_json(), file=output_buffer)

    output_buffer.seek(0)
    await write_file(args.output_file, output_buffer.read().strip())


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Aphrodite API server version {APHRODITE_VERSION}")
    logger.info(f"args: {args}")

    asyncio.run(main(args))
