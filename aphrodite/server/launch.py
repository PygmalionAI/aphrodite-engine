import asyncio
import os
import signal
from http import HTTPStatus
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from loguru import logger

from aphrodite.common.utils import in_windows
from aphrodite.engine.async_aphrodite import AsyncEngineDeadError
from aphrodite.engine.protocol import AsyncEngineClient

APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH = bool(os.getenv(
    "APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH", 0))


async def serve_http(app: FastAPI, engine: AsyncEngineClient,
                     **uvicorn_kwargs: Any):

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server, engine)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    if in_windows():
        # Windows - use signal.signal() directly
        signal.signal(signal.SIGINT, lambda signum, frame: signal_handler())
        signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler())
    else:
        # Unix - use asyncio's add_signal_handler
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("Gracefully stopping http server")
        return server.shutdown()


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server,
                           engine: AsyncEngineClient) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_, __):
        """On generic runtime error, check to see if the engine has died.
        It probably has, in which case the server will no longer be able to
        handle requests. Trigger a graceful shutdown with a SIGTERM."""
        if (not APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH and engine.errored
                and not engine.is_running):
            logger.error("AsyncAphrodite has failed, terminating server "
                         "process")
            # See discussions here on shutting down a uvicorn server
            # https://github.com/encode/uvicorn/discussions/1103
            # In this case we cannot await the server shutdown here because
            # this handler must first return to close the connection for
            # this request.
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @app.exception_handler(AsyncEngineDeadError)
    async def engine_dead_handler(_, __):
        """Kill the server if the async engine is already dead. It will
        not handle any further requests."""
        if not APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH:
            logger.error("AsyncAphrodite is already dead, terminating server "
                         "process")
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
