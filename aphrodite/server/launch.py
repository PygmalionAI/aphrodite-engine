import asyncio
import signal
from http import HTTPStatus
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from loguru import logger

import aphrodite.common.envs as envs
from aphrodite.common.utils import find_process_using_port, in_windows
from aphrodite.engine.async_aphrodite import AsyncEngineDeadError

APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH = (
    envs.APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH)


async def serve_http(app: FastAPI, limit_concurrency: Optional[int],
                     **uvicorn_kwargs: Any):

    # Set concurrency limits in uvicorn if running in multiprocessing mode
    # since zmq has maximum socket limit of zmq.constants.SOCKET_LIMIT (65536).
    if limit_concurrency is not None:
        logger.info(
            "Launching Uvicorn with --limit_concurrency "
            f"{limit_concurrency}. "
            f"To avoid this limit at the expense of performance run with "
            "--disable-frontend-multiprocessing", limit_concurrency)
        uvicorn_kwargs["limit_concurrency"] = limit_concurrency

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

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
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.info(
                f"port {port} is used by process {process} launched with "
                f"command:\n{' '.join(process.cmdline())}")
        logger.info("Gracefully stopping http server")
        return server.shutdown()


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, __):
        """On generic runtime error, check to see if the engine has died.
        It probably has, in which case the server will no longer be able to
        handle requests. Trigger a graceful shutdown with a SIGTERM."""
        engine = request.app.state.engine_client
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
