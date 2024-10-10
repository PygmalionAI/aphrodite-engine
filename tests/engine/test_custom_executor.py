import asyncio
import os

import pytest

from aphrodite.common.sampling_params import SamplingParams
from aphrodite.engine.args_tools import AsyncEngineArgs, EngineArgs
from aphrodite.engine.async_aphrodite import AphroditeEngine, AsyncAphrodite
from aphrodite.executor.gpu_executor import GPUExecutor, GPUExecutorAsync


class Mock:
    ...


class CustomGPUExecutor(GPUExecutor):

    def execute_model(self, *args, **kwargs):
        # Drop marker to show that this was ran
        with open(".marker", "w"):
            ...
        return super().execute_model(*args, **kwargs)


class CustomGPUExecutorAsync(GPUExecutorAsync):

    async def execute_model_async(self, *args, **kwargs):
        with open(".marker", "w"):
            ...
        return await super().execute_model_async(*args, **kwargs)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_type_checking(model):
    with pytest.raises(ValueError):
        engine_args = EngineArgs(model=model,
                                 distributed_executor_backend=Mock)
        AphroditeEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(model=model,
                                      distributed_executor_backend=Mock)
        AsyncAphrodite.from_engine_args(engine_args)
    with pytest.raises(TypeError):
        engine_args = AsyncEngineArgs(
            model=model, distributed_executor_backend=CustomGPUExecutor)
        AsyncAphrodite.from_engine_args(engine_args)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=model, distributed_executor_backend=CustomGPUExecutor)
        engine = AphroditeEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_custom_executor_async(model, tmpdir):
    cwd = os.path.abspath(".")
    os.chdir(tmpdir)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=model, distributed_executor_backend=CustomGPUExecutorAsync)
        engine = AsyncAphrodite.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = await engine.add_request("0", "foo", sampling_params)
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
