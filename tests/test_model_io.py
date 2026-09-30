"""Tests for the model envelope (provenance + compatibility guard + container format)."""

from __future__ import annotations

import json
import pickle
import sys
import types
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from eo_inference_toolbox.eocore import model_io
from eo_inference_toolbox.eocore.model_io import (
    ENVELOPE_VERSION,
    check_compatible,
    load_model,
    model_envelope,
    save_model,
)


def _pickle_to(path: Path, obj: object) -> None:
    """Write ``obj`` straight to ``path`` as a pickle (a legacy pre-container envelope)."""
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _write_container(path: Path, meta: Mapping[str, Any], members: Mapping[str, bytes]) -> None:
    """Hand-build a zip container with a JSON ``meta`` member plus arbitrary members."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("envelope.json", json.dumps(meta))
        for name, blob in members.items():
            zf.writestr(name, blob)


def test_envelope_roundtrips(tmp_path: Path) -> None:
    env = model_envelope(
        {"weights": [1, 2, 3]},
        backend_id="clay",
        n_bands=768,
        kind="classifier",
        classes=["a", "b"],
        extra={"n_estimators": 100},
    )
    assert env["envelope_version"] == ENVELOPE_VERSION
    path = tmp_path / "model.pkl"
    save_model(path, env)
    loaded = load_model(path)
    assert loaded["backend_id"] == "clay"
    assert loaded["n_bands"] == 768
    assert loaded["classes"] == ["a", "b"]
    assert loaded["extra"] == {"n_estimators": 100}
    assert loaded["estimator"] == {"weights": [1, 2, 3]}


def test_check_compatible_passes_on_match() -> None:
    env = model_envelope(object(), backend_id="clay", n_bands=768, kind="cluster")
    check_compatible(env, backend_id="clay", n_bands=768)


def test_check_compatible_rejects_band_mismatch() -> None:
    env = model_envelope(object(), backend_id="clay", n_bands=768, kind="cluster")
    with pytest.raises(ValueError, match="embedding bands"):
        check_compatible(env, backend_id="clay", n_bands=1024)


def test_check_compatible_rejects_backend_mismatch() -> None:
    env = model_envelope(object(), backend_id="clay", n_bands=768, kind="cluster")
    with pytest.raises(ValueError, match="trained on backend"):
        check_compatible(env, backend_id="prithvi", n_bands=768)


def test_check_compatible_tolerates_unknown_backend() -> None:
    env = model_envelope(object(), backend_id="clay", n_bands=768, kind="cluster")
    # A None current backend id skips the backend check (only bands enforced).
    check_compatible(env, backend_id=None, n_bands=768)


def test_load_model_rejects_non_dict(tmp_path: Path) -> None:
    path = tmp_path / "bad.pkl"
    _pickle_to(path, [1, 2, 3])
    with pytest.raises(ValueError, match="valid model envelope"):
        load_model(path)


def test_load_model_rejects_garbage_bytes(tmp_path: Path) -> None:
    # A file that is not pickle data at all must fail closed with a clear message,
    # not surface a raw UnpicklingError to the algorithm layer.
    path = tmp_path / "garbage.pkl"
    path.write_bytes(b"this is definitely not a pickle")
    with pytest.raises(ValueError, match="not readable model data"):
        load_model(path)


def test_load_model_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.pkl"
    path.write_bytes(b"")  # pickle.load raises EOFError here
    with pytest.raises(ValueError, match="not readable model data"):
        load_model(path)


def test_load_model_rejects_dict_missing_envelope_keys(tmp_path: Path) -> None:
    # A plain dict that unpickles fine but is not one of our envelopes: e.g. it
    # lacks the estimator (and version). Fail closed rather than KeyError later.
    path = tmp_path / "partial.pkl"
    _pickle_to(path, {"backend_id": "clay", "n_bands": 768})
    with pytest.raises(ValueError, match="missing key"):
        load_model(path)


def test_load_model_rejects_non_integer_version(tmp_path: Path) -> None:
    env = model_envelope({"w": 1}, backend_id="clay", n_bands=768, kind="classifier")
    env["envelope_version"] = "1"  # a string, not an int
    path = tmp_path / "strver.pkl"
    _pickle_to(path, env)
    with pytest.raises(ValueError, match="non-integer envelope_version"):
        load_model(path)


def test_load_model_rejects_newer_envelope_version(tmp_path: Path) -> None:
    env = model_envelope({"w": 1}, backend_id="clay", n_bands=768, kind="classifier")
    env["envelope_version"] = ENVELOPE_VERSION + 1
    path = tmp_path / "future.pkl"
    _pickle_to(path, env)
    with pytest.raises(ValueError, match="newer model format"):
        load_model(path)


def test_load_model_accepts_current_version_envelope(tmp_path: Path) -> None:
    # The positive case: an envelope at exactly ENVELOPE_VERSION round-trips.
    env = model_envelope({"w": [1, 2]}, backend_id="clay", n_bands=768, kind="classifier")
    path = tmp_path / "ok.pkl"
    save_model(path, env)
    loaded = load_model(path)
    assert loaded["envelope_version"] == ENVELOPE_VERSION
    assert loaded["estimator"] == {"w": [1, 2]}


# --- container format (skops absent -> pickle fallback member) --------------


def test_save_model_writes_zip_container_with_json_metadata(tmp_path: Path) -> None:
    # With skops unavailable (the default in this test env) the estimator is stored
    # in a pickle member, but the envelope is still a zip whose metadata is plain JSON.
    env = model_envelope(
        {"weights": [1, 2, 3]},
        backend_id="clay",
        n_bands=768,
        kind="classifier",
        classes=["a", "b"],
    )
    path = tmp_path / "model.pkl"
    save_model(path, env)

    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        assert "envelope.json" in names
        assert "estimator.pkl" in names  # pickle fallback member
        assert "estimator.skops" not in names
        meta = json.loads(zf.read("envelope.json"))
    # The metadata member is pure JSON and carries NO estimator: reading + validating
    # provenance never unpickles anything (the whole point of the container format).
    assert "estimator" not in meta
    assert meta["serialization"] == "pickle"
    assert meta["backend_id"] == "clay"
    assert meta["envelope_version"] == ENVELOPE_VERSION


def test_container_json_metadata_coerces_numpy_like_values(tmp_path: Path) -> None:
    # Class labels can arrive as numpy scalars; the JSON metadata must still write.
    class _FakeNpScalar:
        def __init__(self, value: int) -> None:
            self._value = value

        def item(self) -> int:
            return self._value

    env = model_envelope(
        {"w": 0},
        backend_id="clay",
        n_bands=4,
        kind="classifier",
        classes=[_FakeNpScalar(3), _FakeNpScalar(7)],
    )
    path = tmp_path / "np.pkl"
    save_model(path, env)
    loaded = load_model(path)
    assert loaded["classes"] == [3, 7]  # coerced to plain ints on write


def test_container_missing_metadata_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "no_meta.pkl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("something_else.txt", b"x")
    with pytest.raises(ValueError, match=r"missing envelope\.json"):
        load_model(path)


def test_container_missing_required_key_is_rejected(tmp_path: Path) -> None:
    meta = {"envelope_version": ENVELOPE_VERSION, "backend_id": "clay", "n_bands": 4}
    path = tmp_path / "partial.pkl"  # 'kind' is absent
    _write_container(path, meta, {"estimator.pkl": pickle.dumps({"w": 1})})
    with pytest.raises(ValueError, match="missing key"):
        load_model(path)


def test_container_missing_estimator_member_is_rejected(tmp_path: Path) -> None:
    meta = {
        "envelope_version": ENVELOPE_VERSION,
        "backend_id": "clay",
        "n_bands": 4,
        "kind": "classifier",
        "serialization": "pickle",
    }
    path = tmp_path / "no_est.pkl"  # serialization says pickle, but no estimator.pkl member
    _write_container(path, meta, {})
    with pytest.raises(ValueError, match="missing key\\(s\\): estimator"):
        load_model(path)


def test_container_unknown_serialization_is_rejected(tmp_path: Path) -> None:
    meta = {
        "envelope_version": ENVELOPE_VERSION,
        "backend_id": "clay",
        "n_bands": 4,
        "kind": "classifier",
        "serialization": "rot13",
    }
    path = tmp_path / "weird.pkl"
    _write_container(path, meta, {"estimator.pkl": pickle.dumps({"w": 1})})
    with pytest.raises(ValueError, match="unknown estimator serialization"):
        load_model(path)


def test_container_invalid_json_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "badjson.pkl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("envelope.json", b"{ this is not json")
    with pytest.raises(ValueError, match="unreadable envelope metadata"):
        load_model(path)


def test_container_metadata_must_be_json_object(tmp_path: Path) -> None:
    path = tmp_path / "array.pkl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("envelope.json", b"[1, 2, 3]")  # valid JSON, wrong shape
    with pytest.raises(ValueError, match="not an object"):
        load_model(path)


def test_container_newer_version_is_rejected(tmp_path: Path) -> None:
    meta = {
        "envelope_version": ENVELOPE_VERSION + 1,
        "backend_id": "clay",
        "n_bands": 4,
        "kind": "classifier",
        "serialization": "pickle",
    }
    path = tmp_path / "future_container.pkl"
    _write_container(path, meta, {"estimator.pkl": pickle.dumps({"w": 1})})
    with pytest.raises(ValueError, match="newer model format"):
        load_model(path)


# --- skops estimator format (exercised via an injected fake skops.io) --------


def _install_fake_skops(
    monkeypatch: pytest.MonkeyPatch, untrusted: Sequence[str]
) -> dict[str, Any]:
    """Inject a fake ``skops.io`` so the skops save/load branch runs without real skops.

    The fake stores a marker + pickle so bytes round-trip in-sandbox; it records the
    ``trusted`` list load_model passes so the allowlist logic itself can be asserted.
    This tests our dispatch + trusted-types gate, not skops' own (de)serialization.
    """
    calls: dict[str, Any] = {}

    def dumps(obj: Any) -> bytes:
        calls["dumped"] = obj
        return b"SKOPS:" + pickle.dumps(obj)

    def get_untrusted_types(*, data: bytes) -> list[str]:
        calls["scanned"] = data
        return list(untrusted)

    def loads(data: bytes, *, trusted: Sequence[str]) -> Any:
        calls["trusted"] = list(trusted)
        assert data.startswith(b"SKOPS:")
        return pickle.loads(data[len(b"SKOPS:") :])

    fake_io = types.ModuleType("skops.io")
    fake_io.dumps = dumps  # type: ignore[attr-defined]
    fake_io.get_untrusted_types = get_untrusted_types  # type: ignore[attr-defined]
    fake_io.loads = loads  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("skops")
    fake_pkg.io = fake_io  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skops", fake_pkg)
    monkeypatch.setitem(sys.modules, "skops.io", fake_io)
    monkeypatch.setattr(model_io, "_skops_available", lambda: True)
    return calls


def test_skops_roundtrip_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_skops(
        monkeypatch, untrusted=["sklearn.ensemble._forest.RandomForestClassifier", "numpy.ndarray"]
    )
    env = model_envelope({"weights": [4, 5, 6]}, backend_id="clay", n_bands=768, kind="classifier")
    path = tmp_path / "skops_model.pkl"
    save_model(path, env)

    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("envelope.json"))
    assert "estimator.skops" in names and "estimator.pkl" not in names
    assert meta["serialization"] == "skops"

    loaded = load_model(path)
    assert loaded["estimator"] == {"weights": [4, 5, 6]}
    # Every discovered type was in the allowlist, so all were passed as trusted.
    assert calls["trusted"] == [
        "sklearn.ensemble._forest.RandomForestClassifier",
        "numpy.ndarray",
    ]


def test_skops_load_rejects_untrusted_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A type outside the sklearn/numpy/scipy allowlist must abort the load, so a
    # hostile file cannot smuggle an arbitrary callable through the skops path.
    _install_fake_skops(monkeypatch, untrusted=["numpy.ndarray", "posix.system"])
    env = model_envelope({"w": 1}, backend_id="clay", n_bands=4, kind="classifier")
    path = tmp_path / "hostile.pkl"
    save_model(path, env)
    with pytest.raises(ValueError, match="outside the trusted"):
        load_model(path)


def test_skops_payload_without_skops_installed_fails_closed(tmp_path: Path) -> None:
    # A container written with skops, opened where skops is absent (this sandbox):
    # fail closed with a clear install hint rather than a confusing import error.
    meta = {
        "envelope_version": ENVELOPE_VERSION,
        "backend_id": "clay",
        "n_bands": 4,
        "kind": "classifier",
        "serialization": "skops",
    }
    path = tmp_path / "needs_skops.pkl"
    _write_container(path, meta, {"estimator.skops": b"SKOPS:whatever"})
    assert model_io._skops_available() is False  # precondition for this sandbox
    with pytest.raises(ValueError, match=r"skops.*not installed"):
        load_model(path)


def test_legacy_raw_pickle_still_loads(tmp_path: Path) -> None:
    # A model saved by an older build (a bare pickle, not a zip) must still load.
    env = model_envelope({"w": [9]}, backend_id="clay", n_bands=768, kind="classifier")
    env["envelope_version"] = 1  # what the old build stamped
    path = tmp_path / "legacy.pkl"
    _pickle_to(path, env)
    assert not zipfile.is_zipfile(path)
    loaded = load_model(path)
    assert loaded["estimator"] == {"w": [9]}
    assert loaded["envelope_version"] == 1
