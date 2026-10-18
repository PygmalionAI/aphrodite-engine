# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def dummy_platform_plugin() -> str | None:
    # Real platform plugins often import Aphrodite modules while registering.
    from aphrodite.utils.torch_utils import direct_register_custom_op  # noqa: F401

    return "aphrodite_add_dummy_platform.dummy_platform.DummyPlatform"


def register_ops():
    import aphrodite_add_dummy_platform.dummy_custom_ops  # noqa
