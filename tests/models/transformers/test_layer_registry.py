# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers backend's hw-agnostic layer resolution.

`layers._resolve` imports a layer symbol from
`aphrodite.model_executor.hw_agnostic.layers.<module>` when `APHRODITE_USE_HW_AGNOSTIC`
is set and the symbol exists, and otherwise falls back to
`aphrodite.model_executor.layers.<module>`. These tests pin that contract and the
logging that reports which source was used.

CPU-only: no engine is built. The end-to-end counterpart that serves a model
through `aphrodite_runner` lives in `test_layer_registry_e2e.py` (GPU job).
"""

import logging
import sys
import types

import pytest

from aphrodite.model_executor.models.transformers import layers

HW_MODULE = "aphrodite.model_executor.hw_agnostic.layers.layernorm"


@pytest.fixture
def fake_hw_layernorm(monkeypatch):
    """Inject a hw-agnostic `layernorm` module exposing a sentinel `RMSNorm`.

    A `SimpleNamespace` stands in for the module: `importlib.import_module`
    returns it from `sys.modules` and `getattr` resolves `RMSNorm`, while its
    attributes are set at construction (no `ModuleType` attribute-set that mypy
    rejects, no constant `setattr` that ruff rejects)."""
    module = types.SimpleNamespace(RMSNorm=type("HwRMSNorm", (), {}))
    monkeypatch.setitem(sys.modules, HW_MODULE, module)
    return module


def test_falls_back_to_aphrodite_when_disabled(monkeypatch, fake_hw_layernorm):
    """Disabled: the Aphrodite class is used even if a hw-agnostic one exists."""
    monkeypatch.setenv("APHRODITE_USE_HW_AGNOSTIC", "0")
    from aphrodite.model_executor.layers.layernorm import RMSNorm as AphroditeRMSNorm

    assert layers._resolve("layernorm", "RMSNorm") is AphroditeRMSNorm


def test_uses_hw_agnostic_when_enabled(monkeypatch, fake_hw_layernorm, caplog_aphrodite):
    """Enabled and available: the hw-agnostic class is used and logged."""
    monkeypatch.setenv("APHRODITE_USE_HW_AGNOSTIC", "1")
    with caplog_aphrodite.at_level(logging.INFO):
        resolved = layers._resolve("layernorm", "RMSNorm")
    assert resolved is fake_hw_layernorm.RMSNorm
    assert "Using hw-agnostic layer: RMSNorm" in caplog_aphrodite.text


def test_falls_back_when_symbol_missing(monkeypatch, caplog_aphrodite):
    """Enabled but the symbol is not ported: fall back to Aphrodite and warn."""
    monkeypatch.setenv("APHRODITE_USE_HW_AGNOSTIC", "1")
    # A hw-agnostic module without the requested attribute triggers fallback.
    empty = types.ModuleType(HW_MODULE)
    monkeypatch.setitem(sys.modules, HW_MODULE, empty)
    from aphrodite.model_executor.layers.layernorm import RMSNorm as AphroditeRMSNorm

    with caplog_aphrodite.at_level(logging.WARNING):
        resolved = layers._resolve("layernorm", "RMSNorm")
    assert resolved is AphroditeRMSNorm
    assert "falling back to default" in caplog_aphrodite.text


def test_act_and_mul_falls_back_for_unknown_activation(monkeypatch, default_aphrodite_config):
    """An activation with no hw-agnostic equivalent falls back to Aphrodite's.

    `default_aphrodite_config` supplies the config context the CustomOp needs.
    """
    monkeypatch.setenv("APHRODITE_USE_HW_AGNOSTIC", "1")
    from aphrodite.model_executor.layers.activation import GeluAndMul

    assert isinstance(layers.get_act_and_mul_fn("gelu"), GeluAndMul)
