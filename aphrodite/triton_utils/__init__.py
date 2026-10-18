# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from typing import TYPE_CHECKING, Any

from aphrodite.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

gluon: Any
gl: Any
aggregate: Any

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    gluon = import_module("triton.experimental.gluon")
    gl = import_module("triton.experimental.gluon.language")
    aggregate = import_module("triton.language.core")._aggregate
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()

from aphrodite.triton_utils.tensor_descriptor import use_tensor_descriptor  # noqa: E402

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453


def get_cache_invalidating_env_vars() -> dict[str, str]:
    """Read Triton's cache environment without eagerly importing its compiler."""
    get_env_vars = import_module("triton._C.libtriton").get_cache_invalidating_env_vars

    return get_env_vars()


__all__ = [
    "HAS_TRITON",
    "get_cache_invalidating_env_vars",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "gluon",
    "gl",
    "aggregate",
    "use_tensor_descriptor",
]
