import asyncio
import os
import pickle
import signal
from typing import Any, Coroutine, Union

import cloudpickle
import zmq
import zmq.asyncio
from loguru import logger
from typing_extensions import Never
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from aphrodite import AsyncAphrodite, AsyncEngineArgs
from aphrodite.common.config import (DecodingConfig, LoRAConfig, ModelConfig,
                                     ParallelConfig, SchedulerConfig)
from aphrodite.common.utils import in_windows
from aphrodite.endpoints.openai.rpc import (APHRODITE_RPC_SUCCESS_STR,
                                            APHRODITE_RPC_ZMQ_HWM,
                                            RPCAbortRequest,
                                            RPCGenerateRequest,
                                            RPCUtilityRequest)

if in_windows():
    import winloop as uvloop
else:
    import uvloop


CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]


class AsyncEngineRPCServer:

    def __init__(self, async_engine_args: AsyncEngineArgs, rpc_path: str):
        # Initialize engine first.
        self.engine = AsyncAphrodite.from_engine_args(async_engine_args)

        # Initialize context.
        self.context = zmq.asyncio.Context()

        # Init socket.
        self.socket: Socket = self.context.socket(zmq.constants.DEALER)
        self.socket.set_hwm(APHRODITE_RPC_ZMQ_HWM)
        self.socket.connect(rpc_path)

    def cleanup(self):
        """Cleanup all resources."""
        self.socket.close()
        self.context.destroy()
        self.engine.shutdown_background_loop()
        # Clear the engine reference so that it can be GC'ed.
        self.engine = None

    async def get_config(self, identity, request):
        try:
            config: CONFIG_TYPE
            if request == RPCUtilityRequest.GET_MODEL_CONFIG:
                config = await self.engine.get_model_config()
            elif request == RPCUtilityRequest.GET_DECODING_CONFIG:
                config = await self.engine.get_decoding_config()
            elif request == RPCUtilityRequest.GET_LORA_CONFIG:
                config = await self.engine.get_lora_config()
            elif request == RPCUtilityRequest.GET_SCHEDULER_CONFIG:
                config = await self.engine.get_scheduler_config()
            elif request == RPCUtilityRequest.GET_PARALLEL_CONFIG:
                config = await self.engine.get_parallel_config()
            else:
                raise ValueError(f"Unknown Config Request: {request}")

            await self.socket.send_multipart((identity, pickle.dumps(config)),
                                             copy=False)

        except Exception as e:
            await self.socket.send_multipart((identity, pickle.dumps(e)),
                                             copy=False)

    async def do_log_stats(self, identity):
        """Log stats and confirm success."""
        await self.engine.do_log_stats()

        await self.socket.send_multipart(
            (identity, pickle.dumps(APHRODITE_RPC_SUCCESS_STR)))

    async def is_server_ready(self, identity):
        """Notify the client that we are ready."""
        await self.socket.send_multipart(
            (identity, pickle.dumps(APHRODITE_RPC_SUCCESS_STR)))

    async def abort(self, identity, request: RPCAbortRequest):
        """Abort request and notify the client of success."""
        try:
            # Abort the request in the llm engine.
            await self.engine.abort(request.request_id)
            result: Union[str, Exception] = APHRODITE_RPC_SUCCESS_STR
        except Exception as e:
            result = e
        await self.socket.send_multipart((identity, pickle.dumps(result)))

    async def generate(self, identity, generate_request: RPCGenerateRequest):
        try:
            results_generator = self.engine.generate(
                generate_request.inputs,
                sampling_params=generate_request.sampling_params,
                request_id=generate_request.request_id,
                lora_request=generate_request.lora_request,
                prompt_adapter_request=generate_request.prompt_adapter_request)

            async for request_output in results_generator:
                await self.socket.send_multipart(
                    (identity, pickle.dumps(request_output)), copy=False)

        except Exception as e:
            await self.socket.send_multipart((identity, pickle.dumps(e)),
                                             copy=False)

    async def check_health(self, identity):
        try:
            await self.engine.check_health()
            await self.socket.send_multipart(
                (identity, pickle.dumps(APHRODITE_RPC_SUCCESS_STR)))

        except Exception as e:
            await self.socket.send_multipart((identity, pickle.dumps(e)),
                                             copy=False)

    def _make_handler_coro(self, identity,
                           message: Frame) -> Coroutine[Any, Any, Never]:
        """Route the zmq message to the handler coroutine."""

        request = cloudpickle.loads(message.buffer)

        if isinstance(request, RPCGenerateRequest):
            return self.generate(identity, request)

        elif isinstance(request, RPCAbortRequest):
            return self.abort(identity, request)

        elif isinstance(request, RPCUtilityRequest):
            if request in [
                    RPCUtilityRequest.GET_MODEL_CONFIG,
                    RPCUtilityRequest.GET_PARALLEL_CONFIG,
                    RPCUtilityRequest.GET_DECODING_CONFIG,
                    RPCUtilityRequest.GET_SCHEDULER_CONFIG,
                    RPCUtilityRequest.GET_LORA_CONFIG
            ]:
                return self.get_config(identity, request)
            elif request == RPCUtilityRequest.DO_LOG_STATS:
                return self.do_log_stats(identity)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                return self.is_server_ready(identity)
            elif request == RPCUtilityRequest.IS_SERVER_HEALTHY:
                return self.check_health(identity)
            elif request == RPCUtilityRequest.SHUTDOWN_SERVER:
                return self.shutdown(identity)
            else:
                raise ValueError(f"Unknown RPCUtilityRequest type: {request}")

        else:
            raise ValueError(f"Unknown RPCRequest type: {request}")

    async def run_server_loop(self):
        """Inner RPC Server Loop"""

        running_tasks = set()
        while True:
            # Wait for a request.
            identity, message = await self.socket.recv_multipart(copy=False)

            # Process the request async.
            task = asyncio.create_task(
                self._make_handler_coro(identity, message))

            # We need to keep around a strong reference to the task,
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)

    async def shutdown(self, identity):
        """Handle shutdown request from client."""
        try:
            # Clean shutdown of engine
            self.engine.shutdown_background_loop()
            await self.socket.send_multipart(
                [identity, cloudpickle.dumps(APHRODITE_RPC_SUCCESS_STR)]
            )
        except Exception as e:
            await self.socket.send_multipart([identity, cloudpickle.dumps(e)])
        finally:
            # Schedule server shutdown
            asyncio.create_task(self._delayed_shutdown())
    
    async def _delayed_shutdown(self):
        """Helper to shut down server after response is sent"""
        await asyncio.sleep(1)
        self.cleanup()
        # Force exit the process
        os._exit(0)



async def run_server(server: AsyncEngineRPCServer):
    # Put the server task into the asyncio loop.
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.run_server_loop())

    # Interruption handling.
    def signal_handler() -> None:
        # Kill the server on interrupt / terminate
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Aphrodite ZMQ RPC Server was interrupted.")
    finally:
        # Clean up all resources.
        server.cleanup()


def run_rpc_server(async_engine_args: AsyncEngineArgs, rpc_path: str):
    server = AsyncEngineRPCServer(async_engine_args, rpc_path)
    uvloop.run(run_server(server))
