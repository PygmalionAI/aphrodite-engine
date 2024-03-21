import pytest
import torch

from aphrodite.modeling.layers.sampler import _apply_typical_sampling

def test_typical_sampling_shape():
    logits = torch.randn(10, 5)
    typical_p = torch.randn(10)
    typical_p_sigma = torch.randn(10)
    output = _apply_typical_sampling(logits, typical_p, typical_p_sigma)
    assert output.shape == logits.shape, "Output shape should match input shape"

def test_typical_sampling_dtype():
    logits = torch.randn(10, 5)
    typical_p = torch.randn(10)
    typical_p_sigma = torch.randn(10)
    output = _apply_typical_sampling(logits, typical_p, typical_p_sigma)
    assert output.dtype == logits.dtype, "Output dtype should match input dtype"

def test_typical_sampling_device():
    logits = torch.randn(10, 5)
    typical_p = torch.randn(10)
    typical_p_sigma = torch.randn(10)
    output = _apply_typical_sampling(logits, typical_p, typical_p_sigma)
    assert output.device == logits.device, "Output dev should match input dev"

def test_typical_sampling_inf():
    logits = torch.randn(10, 5)
    typical_p = torch.randn(10)
    typical_p_sigma = torch.randn(10)
    output = _apply_typical_sampling(logits, typical_p, typical_p_sigma)
    assert not torch.isinf(output).any(), "Output should not contain inf"

def test_typical_sampling_nan():
    logits = torch.randn(10, 5)
    typical_p = torch.randn(10)
    typical_p_sigma = torch.randn(10)
    output = _apply_typical_sampling(logits, typical_p, typical_p_sigma)
    assert not torch.isnan(output).any(), "Output should not contain NaN"
