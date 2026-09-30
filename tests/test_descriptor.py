"""Tests for the embedding descriptor and its validation."""

from __future__ import annotations

import dataclasses

import pytest

from eo_inference_toolbox.eocore.descriptor import (
    DEFAULT_NODATA,
    BackendCapabilities,
    EmbeddingDescriptor,
)


def test_descriptor_defaults() -> None:
    desc = EmbeddingDescriptor(backend_id="clay", n_bands=768)
    assert desc.is_unit_normalized is False
    assert desc.native_dtype == "float32"
    assert desc.nodata == DEFAULT_NODATA
    assert desc.temporal is False
    assert isinstance(desc.capabilities, BackendCapabilities)


def test_descriptor_rejects_empty_backend_id() -> None:
    with pytest.raises(ValueError, match="backend_id"):
        EmbeddingDescriptor(backend_id="", n_bands=10)


def test_descriptor_rejects_nonpositive_bands() -> None:
    with pytest.raises(ValueError, match="n_bands"):
        EmbeddingDescriptor(backend_id="clay", n_bands=0)


def test_descriptor_is_frozen() -> None:
    desc = EmbeddingDescriptor(backend_id="clay", n_bands=32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        desc.n_bands = 64  # type: ignore[misc]


def test_capabilities_defaults() -> None:
    caps = BackendCapabilities()
    assert caps.needs_gpu is False
    assert caps.needs_auth is False
    assert caps.extra_deps == ()
