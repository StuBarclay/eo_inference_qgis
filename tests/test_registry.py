"""Tests for the pluggable-backend registry (E-10) and model-API guards (H-6).

Everything here is stdlib + NumPy only: registration and the guard helpers are
torch-free, and the Clay ``_encoder`` seam is exercised with a fake model object,
so no torch, GDAL or QGIS is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from eo_inference_toolbox.eocore.backends import (
    BACKENDS,
    BackendInfo,
    EmbeddingBackend,
    TileMeta,
    load_backend,
    register_backend,
    registered_backends,
)
from eo_inference_toolbox.eocore.backends import base as _base
from eo_inference_toolbox.eocore.backends.base import (
    require_model_attr,
    select_feature,
    unsupported_model_error,
)


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


def _make_info(backend_id: str) -> BackendInfo:
    return BackendInfo(
        backend_id=backend_id,
        display_name=f"Fake {backend_id}",
        summary="A test-only backend.",
        default_bands=("red", "green", "blue"),
        default_collection="test",
        embedding_dim=8,
        resolution_m=10.0,
        is_unit_normalized=False,
        licence="test",
        pip_packages=("torch",),
        weights_hint="none",
    )


class _FakeBackend:
    """A minimal structural :class:`EmbeddingBackend`."""

    backend_id = "fake"
    display_name = "Fake"

    def descriptor(self) -> Any:  # pragma: no cover - not exercised
        raise NotImplementedError

    def required_bands(self) -> tuple[str, ...]:
        return ("red", "green", "blue")

    def embed_tile(self, image: Any, meta: TileMeta) -> Any:
        return image


# -- E-10 registration --------------------------------------------------------
def test_builtin_backends_are_registered() -> None:
    assert {"clay", "prithvi", "satlas"} <= set(registered_backends())


def test_register_backend_is_reachable_via_load_backend(registry_sandbox: None) -> None:
    register_backend(_make_info("fakeload"), lambda **cfg: _FakeBackend())
    assert "fakeload" in registered_backends()
    assert BACKENDS["fakeload"].display_name == "Fake fakeload"
    obj = load_backend("fakeload")
    assert isinstance(obj, _FakeBackend)
    assert isinstance(obj, EmbeddingBackend)  # structural conformance


def test_registration_is_lazy_loader_not_called_until_load(registry_sandbox: None) -> None:
    calls: list[dict[str, Any]] = []

    def loader(**config: Any) -> _FakeBackend:
        calls.append(config)
        return _FakeBackend()

    register_backend(_make_info("lazyfake"), loader)
    assert calls == []  # registering must not call the loader (stays torch-free)
    load_backend("lazyfake", device="cpu", batch=4)
    assert calls == [{"device": "cpu", "batch": 4}]  # config forwarded verbatim


def test_register_duplicate_raises_unless_replace(registry_sandbox: None) -> None:
    register_backend(_make_info("dupfake"), lambda **cfg: _FakeBackend())
    with pytest.raises(ValueError, match="already registered"):
        register_backend(_make_info("dupfake"), lambda **cfg: _FakeBackend())
    # replace=True overrides in place (no error).
    register_backend(_make_info("dupfake"), lambda **cfg: _FakeBackend(), replace=True)


def test_load_backend_unknown_lists_registered() -> None:
    with pytest.raises(KeyError, match="unknown backend"):
        load_backend("no-such-backend")


# -- H-6 guard helpers --------------------------------------------------------
def test_unsupported_model_error_is_actionable() -> None:
    err = unsupported_model_error("clay", "the model exposes no encoder")
    assert isinstance(err, RuntimeError)
    message = str(err)
    assert "clay" in message
    assert "the model exposes no encoder" in message
    assert "checkpoint or model-library version" in message
    assert "SMOKE_TEST" in message


def test_require_model_attr_returns_present_attribute() -> None:
    class Obj:
        encoder = "the-encoder"

    assert require_model_attr(Obj(), "encoder", "clay", "no encoder") == "the-encoder"


def test_require_model_attr_missing_raises_actionable() -> None:
    class Obj:
        pass

    with pytest.raises(RuntimeError, match="model-library version"):
        require_model_attr(Obj(), "encoder", "clay", "no '.encoder'")


def test_require_model_attr_none_value_raises() -> None:
    class Obj:
        encoder = None

    with pytest.raises(RuntimeError):
        require_model_attr(Obj(), "encoder", "prithvi", "encoder is None")


def test_select_feature_passthrough_and_indexing() -> None:
    sentinel = object()
    assert select_feature(sentinel, "clay", index=0) is sentinel  # non-sequence: as-is
    first, last = object(), object()
    assert select_feature([first, last], "prithvi", index=-1) is last
    assert select_feature((first, last), "clay", index=0) is first


def test_select_feature_empty_sequence_raises() -> None:
    with pytest.raises(RuntimeError, match="no feature tensors"):
        select_feature([], "satlas", index=0)
    with pytest.raises(RuntimeError, match="no feature tensors"):
        select_feature((), "prithvi", index=-1)


# -- H-6 Clay encoder seam (fake model, no torch) -----------------------------
class _Encoder:
    def __call__(self, cube: Any) -> Any:  # pragma: no cover - not called here
        return cube


class _ModelWithInner:
    """Clay's usual shape: ``model.model.encoder``."""

    def __init__(self) -> None:
        self.model = _Inner()


class _Inner:
    def __init__(self) -> None:
        self.encoder = _Encoder()


class _ModelDirectEncoder:
    """A checkpoint that exposes ``.encoder`` directly (no ``.model``)."""

    def __init__(self) -> None:
        self.encoder = _Encoder()


class _ModelNoEncoder:
    """A model whose expected encoder attribute is missing."""


def test_clay_encoder_finds_nested_encoder() -> None:
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    backend = ClayBackend()
    backend._model = _ModelWithInner()
    assert isinstance(backend._encoder(), _Encoder)


def test_clay_encoder_falls_back_to_model_itself() -> None:
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    backend = ClayBackend()
    backend._model = _ModelDirectEncoder()
    assert isinstance(backend._encoder(), _Encoder)


def test_clay_encoder_missing_raises_actionable() -> None:
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    backend = ClayBackend()
    backend._model = _ModelNoEncoder()
    with pytest.raises(RuntimeError, match="checkpoint or model-library version"):
        backend._encoder()
