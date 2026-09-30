"""Registry-walk conformance tests for every backend (H-14 / E-13).

For each *registered* backend this asserts two things without importing torch,
GDAL or QGIS:

1. the constructed backend satisfies the :class:`EmbeddingBackend` protocol
   (construction is torch-free — the heavy imports live inside ``embed_tile``);
2. its :class:`BackendInfo` is internally self-consistent (positive dimension
   and resolution, non-empty band list, a declared ``torch`` dependency, and a
   descriptor that agrees with the registered facts).

A malformed new backend — one that violates the protocol or carries inconsistent
metadata — therefore fails at CI time rather than at a user's first embed run.
The negative tests at the bottom prove the walk actually rejects such backends.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from eo_inference_toolbox.eocore.backends import (
    BackendInfo,
    EmbeddingBackend,
    TileMeta,
    backend_info,
    load_backend,
    register_backend,
    registered_backends,
)
from eo_inference_toolbox.eocore.backends import base as _base
from eo_inference_toolbox.eocore.descriptor import EmbeddingDescriptor


@pytest.fixture
def registry_sandbox() -> Iterator[None]:
    """Snapshot the registry and drop any backend ids added during the test."""
    before = set(_base._BACKEND_LOADERS)
    try:
        yield
    finally:
        for backend_id in list(_base._BACKEND_LOADERS):
            if backend_id not in before:
                _base._BACKEND_LOADERS.pop(backend_id, None)
                _base.BACKENDS.pop(backend_id, None)


def assert_backend_conforms(backend_id: str) -> None:
    """Assert a registered backend satisfies the protocol and has consistent facts.

    Raises :class:`AssertionError` (so it fails a test) if the backend violates
    the :class:`EmbeddingBackend` protocol or its :class:`BackendInfo` is
    internally inconsistent. Construction is torch-free, so this runs anywhere.
    """
    info = backend_info(backend_id)

    # -- BackendInfo self-consistency -----------------------------------------
    assert info.backend_id == backend_id, "registry key must match info.backend_id"
    assert info.display_name, "display_name must be non-empty"
    assert info.summary, "summary must be non-empty"
    assert info.licence, "licence must be non-empty"
    assert info.weights_hint, "weights_hint must be non-empty"
    assert info.embedding_dim > 0, "embedding_dim must be positive"
    assert info.resolution_m > 0, "resolution_m must be positive"
    assert info.default_input_scale > 0, "default_input_scale must be positive"
    assert info.default_bands, "default_bands must be non-empty"
    assert all(isinstance(band, str) and band for band in info.default_bands)
    assert info.default_collection, "default_collection must be non-empty"
    assert info.pip_packages, "pip_packages must be non-empty"
    assert all(isinstance(pkg, str) and pkg for pkg in info.pip_packages)
    assert "torch" in info.pip_packages, "every model backend depends on torch"
    assert info.import_names, "import_names must be non-empty (dependency detection)"
    assert all(isinstance(name, str) and name for name in info.import_names)
    # Backends must emit non-unit vectors; the embed algorithm L2-normalises.
    assert info.is_unit_normalized is False, "backends must not pre-normalise"

    # -- protocol conformance (construction is torch-free) --------------------
    backend = load_backend(backend_id)
    assert isinstance(backend, EmbeddingBackend), "must satisfy EmbeddingBackend"
    assert backend.backend_id == backend_id, "instance backend_id must match id"
    assert isinstance(backend.display_name, str) and backend.display_name

    bands = backend.required_bands()
    assert isinstance(bands, tuple) and bands, "required_bands must be a non-empty tuple"
    assert all(isinstance(band, str) and band for band in bands)

    # -- descriptor agrees with the registered facts --------------------------
    descriptor = backend.descriptor()
    assert isinstance(descriptor, EmbeddingDescriptor)
    assert descriptor.backend_id == backend_id
    assert descriptor.n_bands == info.embedding_dim, "nominal band count must match info"
    assert descriptor.n_bands > 0
    assert descriptor.is_unit_normalized == info.is_unit_normalized


@pytest.mark.parametrize("backend_id", registered_backends())
def test_registered_backend_conforms(backend_id: str) -> None:
    assert_backend_conforms(backend_id)


def test_registry_is_non_empty() -> None:
    # Guards against a walk that vacuously "passes" over an empty registry.
    assert registered_backends(), "at least the three built-in backends must register"
    assert {"clay", "prithvi", "satlas"} <= set(registered_backends())


# -- the walk must actually reject a bad backend ------------------------------
def _make_info(backend_id: str, **overrides: Any) -> BackendInfo:
    fields: dict[str, Any] = {
        "backend_id": backend_id,
        "display_name": f"Fake {backend_id}",
        "summary": "A test-only backend.",
        "default_bands": ("red", "green", "blue"),
        "default_collection": "test",
        "embedding_dim": 8,
        "resolution_m": 10.0,
        "is_unit_normalized": False,
        "licence": "test",
        "pip_packages": ("torch",),
        "weights_hint": "none",
        "import_names": ("faker",),
    }
    fields.update(overrides)
    return BackendInfo(**fields)


class _ConformingBackend:
    """A minimal, fully-conforming :class:`EmbeddingBackend`."""

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.display_name = f"Fake {backend_id}"

    def descriptor(self) -> EmbeddingDescriptor:
        return EmbeddingDescriptor(backend_id=self.backend_id, n_bands=8)

    def required_bands(self) -> tuple[str, ...]:
        return ("red", "green", "blue")

    def embed_tile(self, image: Any, meta: TileMeta) -> Any:
        return image


class _ProtocolViolator:
    """Looks like a backend but is missing ``embed_tile`` — not an EmbeddingBackend."""

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.display_name = "Broken"

    def descriptor(self) -> EmbeddingDescriptor:  # pragma: no cover - never reached
        return EmbeddingDescriptor(backend_id=self.backend_id, n_bands=8)

    def required_bands(self) -> tuple[str, ...]:  # pragma: no cover - never reached
        return ("red",)


def test_walk_rejects_protocol_violation(registry_sandbox: None) -> None:
    # The loader intentionally returns a non-conforming object (no embed_tile),
    # which is exactly what the conformance walk must catch; the type-ignore
    # marks that we are feeding register_backend a deliberately-bad loader.
    register_backend(
        _make_info("badproto"),
        lambda **cfg: _ProtocolViolator("badproto"),  # type: ignore[arg-type]
    )
    with pytest.raises(AssertionError, match="EmbeddingBackend"):
        assert_backend_conforms("badproto")


def test_walk_rejects_inconsistent_metadata(registry_sandbox: None) -> None:
    # A conforming object, but its facts are inconsistent (no bands declared).
    register_backend(
        _make_info("badmeta", default_bands=()),
        lambda **cfg: _ConformingBackend("badmeta"),
    )
    with pytest.raises(AssertionError, match="default_bands"):
        assert_backend_conforms("badmeta")


def test_walk_rejects_missing_torch_dependency(registry_sandbox: None) -> None:
    register_backend(
        _make_info("notorch", pip_packages=("numpy",)),
        lambda **cfg: _ConformingBackend("notorch"),
    )
    with pytest.raises(AssertionError, match="torch"):
        assert_backend_conforms("notorch")


def test_walk_accepts_a_well_formed_backend(registry_sandbox: None) -> None:
    register_backend(_make_info("goodfake"), lambda **cfg: _ConformingBackend("goodfake"))
    assert_backend_conforms("goodfake")  # must not raise
