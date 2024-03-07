# Adapted from https://github.com/ray-project/ray/blob/f92928c9cfcbbf80c3a8534ca4911de1b44069c0/python/ray/__init__.py#L11
def _configure_system():
    import os
    import sys

    # Importing flash-attn.
    thirdparty_files = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                    "thirdparty_files")
    sys.path.insert(0, thirdparty_files)


_configure_system()
# Delete configuration function.
del _configure_system

from aphrodite.engine.args_tools import AsyncEngineArgs, EngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.engine.aphrodite_engine import AphroditeEngine
from aphrodite.engine.ray_tools import initialize_cluster
from aphrodite.endpoints.llm import LLM
from aphrodite.common.outputs import CompletionOutput, RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.modeling.layers.attention import Attention

__version__ = "0.4.9"

__all__ = [
    "Attention",
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "AphroditeEngine",
    "EngineArgs",
    "AsyncAphrodite",
    "AsyncEngineArgs",
    "initialize_cluster",
]
