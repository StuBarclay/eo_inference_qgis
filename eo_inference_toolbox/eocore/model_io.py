"""Persist a trained estimator with enough provenance to reuse it safely.

The envelope stamps the number of embedding bands *and the backend id*, so a
model trained on (say) Clay vectors cannot be silently reused on Prithvi vectors
— the dimensions might even match, but the vector spaces do not.

On-disk format
--------------
A saved model is a small **zip container** (written to the user's chosen path,
conventionally ``*.pkl``) holding two members:

* ``envelope.json`` — the provenance metadata (band count, backend id, kind,
  class list, ...) as **plain JSON**, plus a ``serialization`` field naming how
  the estimator was stored; and
* ``estimator.skops`` **or** ``estimator.pkl`` — the fitted estimator itself.

Because the metadata is JSON, reading and validating a model's provenance never
executes code. Older builds wrote a single raw pickle; :func:`load_model` still
reads those (the "legacy" path) so no previously saved model is orphaned.

Security note — trust boundary
-------------------------------
When the optional :mod:`skops` package is installed, the estimator is stored with
skops and reconstructed on load through skops' *trusted-types* gate: this module
only trusts types under the ``sklearn.`` / ``numpy.`` / ``scipy.`` namespaces (the
machinery our classifiers are built from) and refuses anything else, so a hostile
file cannot smuggle an arbitrary callable past the loader. When skops is **not**
installed, the estimator falls back to :mod:`pickle`; legacy files are pickle too.
**Unpickling can execute arbitrary code**, so for the pickle paths the same rule
as before applies: only load model files you trained yourself or trust. The JSON
metadata is always validated fail-closed regardless of the estimator format.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Bumped to 2 when the on-disk format became a zip container (JSON metadata +
#: separate estimator member). Legacy raw-pickle envelopes are version 1 and are
#: still accepted on load; anything newer than this build is rejected.
ENVELOPE_VERSION = 2

#: Keys every envelope carries. :func:`load_model` requires all of them so a file
#: that happens to be a plain ``dict`` (or any other object) is rejected rather
#: than misread as a model. In the container format ``estimator`` lives in its own
#: zip member and the other four keys live in ``envelope.json``.
_REQUIRED_ENVELOPE_KEYS = ("envelope_version", "backend_id", "n_bands", "kind", "estimator")

# Zip container member names.
_META_MEMBER = "envelope.json"
_EST_SKOPS_MEMBER = "estimator.skops"
_EST_PICKLE_MEMBER = "estimator.pkl"

#: When loading a skops payload, only types whose fully-qualified name starts with
#: one of these namespaces may be reconstructed. skops refuses to build any type
#: not explicitly trusted, so this allowlist is the actual code-execution guard: it
#: admits the scikit-learn / numpy / scipy objects our estimators are made of and
#: nothing else (never builtins such as ``posix.system`` or arbitrary callables).
_SKOPS_TRUSTED_PREFIXES = ("sklearn.", "numpy.", "scipy.")


def _skops_available() -> bool:
    """True when the optional :mod:`skops` package can be imported."""
    return importlib.util.find_spec("skops") is not None


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars/arrays (and similar) so metadata stays JSON-serializable.

    Deliberately avoids importing numpy: numpy arrays expose ``.tolist()`` and numpy
    scalars expose both ``.tolist()`` and ``.item()``, so duck-typing keeps this
    module free of a hard numpy dependency while still handling label/extra values
    that arrive as numpy types.
    """
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable in a model envelope")


def model_envelope(
    estimator: Any,
    *,
    backend_id: str,
    n_bands: int,
    kind: str,
    classes: Sequence[Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metadata + estimator dict written to disk."""
    envelope: dict[str, Any] = {
        "envelope_version": ENVELOPE_VERSION,
        "backend_id": backend_id,
        "n_bands": int(n_bands),
        "kind": kind,
        "classes": list(classes) if classes is not None else None,
        "estimator": estimator,
    }
    if extra:
        envelope["extra"] = dict(extra)
    return envelope


def save_model(path: str | Path, envelope: Mapping[str, Any]) -> None:
    """Write ``envelope`` as a zip container (JSON metadata + serialized estimator).

    The estimator is stored with skops when the optional package is installed
    (shrinking the load-time trust boundary, see the module docstring) and with
    pickle otherwise. The metadata is always plain JSON. The ``estimator`` entry is
    pulled out of the envelope and written to its own member; every other key is
    JSON-encoded into ``envelope.json`` along with a ``serialization`` marker.
    """
    meta = dict(envelope)
    estimator = meta.pop("estimator", None)
    if _skops_available():
        import skops.io as sio

        payload: bytes = sio.dumps(estimator)
        estimator_member = _EST_SKOPS_MEMBER
        meta["serialization"] = "skops"
    else:
        payload = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        estimator_member = _EST_PICKLE_MEMBER
        meta["serialization"] = "pickle"
    meta_json = json.dumps(meta, default=_json_default, indent=2, sort_keys=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_META_MEMBER, meta_json)
        zf.writestr(estimator_member, payload)


def load_model(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a model saved by :func:`save_model`.

    Dispatches on the file's shape: a zip container is read through the modern
    JSON-metadata path (:func:`_load_container`); anything else is treated as a
    legacy raw-pickle envelope (:func:`_load_legacy_pickle`). Both fail closed with
    a clear :class:`ValueError` when the file is not readable model data, is missing
    the stamped envelope keys, or was written by a newer format than this build
    understands. A valid envelope produced by :func:`save_model` always passes.

    .. warning::
       For the pickle estimator format (skops absent) and for legacy files, loading
       unpickles the estimator, which **can execute arbitrary code embedded in it**;
       only load model files you trained yourself or trust. The skops format guards
       against this by reconstructing only trusted scikit-learn/numpy/scipy types.
       The JSON metadata is validated without ever executing code either way.
    """
    if zipfile.is_zipfile(path):
        return _load_container(path)
    return _load_legacy_pickle(path)


def _load_container(path: str | Path) -> dict[str, Any]:
    """Read the modern zip container: validate JSON metadata, then the estimator."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if _META_MEMBER not in names:
                raise ValueError(f"{path!r} is not a valid model envelope (missing {_META_MEMBER})")
            try:
                meta_obj: Any = json.loads(zf.read(_META_MEMBER))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path!r} has unreadable envelope metadata: {exc}") from exc
            _validate_meta(meta_obj, path)
            meta: dict[str, Any] = meta_obj
            estimator = _read_estimator(zf, meta, names, path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path!r} is not readable model data: {exc}") from exc
    meta["estimator"] = estimator
    return meta


def _validate_meta(meta: Any, path: str | Path) -> None:
    """Fail closed unless ``meta`` is a dict carrying the required non-estimator keys."""
    if not isinstance(meta, dict):
        got = type(meta).__name__
        raise ValueError(
            f"{path!r} is not a valid model envelope (metadata is {got}, not an object)"
        )
    required = tuple(key for key in _REQUIRED_ENVELOPE_KEYS if key != "estimator")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(
            f"{path!r} is not a valid model envelope (missing key(s): {', '.join(missing)})"
        )
    _check_version(meta["envelope_version"], path)


def _check_version(version: Any, path: str | Path) -> None:
    """Reject a non-integer version or one newer than this build understands."""
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{path!r} has a non-integer envelope_version {version!r}")
    if version > ENVELOPE_VERSION:
        raise ValueError(
            f"{path!r} was written by a newer model format (envelope_version {version} "
            f"> {ENVELOPE_VERSION}); upgrade the plugin to load it"
        )


def _read_estimator(
    zf: zipfile.ZipFile, meta: Mapping[str, Any], names: set[str], path: str | Path
) -> Any:
    """Load the estimator member named by the metadata's ``serialization`` marker."""
    serialization = meta.get("serialization")
    if serialization == "skops":
        if _EST_SKOPS_MEMBER not in names:
            raise ValueError(f"{path!r} is not a valid model envelope (missing key(s): estimator)")
        return _skops_loads(zf.read(_EST_SKOPS_MEMBER), path)
    if serialization == "pickle":
        if _EST_PICKLE_MEMBER not in names:
            raise ValueError(f"{path!r} is not a valid model envelope (missing key(s): estimator)")
        try:
            return pickle.loads(zf.read(_EST_PICKLE_MEMBER))
        except (pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f"{path!r} is not readable model data: {exc}") from exc
    raise ValueError(f"{path!r} has an unknown estimator serialization {serialization!r}")


def _skops_loads(blob: bytes, path: str | Path) -> Any:
    """Deserialize a skops estimator, trusting only allowlisted type namespaces."""
    try:
        import skops.io as sio
    except ImportError as exc:
        raise ValueError(
            f"{path!r} stores its estimator with skops, but the optional 'skops' package "
            "is not installed; install it to load this model"
        ) from exc
    untrusted = list(sio.get_untrusted_types(data=blob))
    disallowed = sorted(t for t in untrusted if not t.startswith(_SKOPS_TRUSTED_PREFIXES))
    if disallowed:
        raise ValueError(
            f"{path!r} contains estimator types outside the trusted "
            f"scikit-learn/numpy/scipy set and will not be loaded: {', '.join(disallowed)}"
        )
    return sio.loads(blob, trusted=untrusted)


def _load_legacy_pickle(path: str | Path) -> dict[str, Any]:
    """Read a pre-container raw-pickle envelope (older builds), validated fail-closed."""
    try:
        with open(path, "rb") as handle:
            obj: Any = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"{path!r} is not readable model data: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(
            f"{path!r} is not a valid model envelope (got {type(obj).__name__}, not a dict)"
        )
    missing = [key for key in _REQUIRED_ENVELOPE_KEYS if key not in obj]
    if missing:
        raise ValueError(
            f"{path!r} is not a valid model envelope (missing key(s): {', '.join(missing)})"
        )
    _check_version(obj["envelope_version"], path)
    return obj


def check_compatible(envelope: Mapping[str, Any], *, backend_id: str | None, n_bands: int) -> None:
    """Raise ``ValueError`` if a saved model does not match the current cube."""
    env_bands = int(envelope.get("n_bands", -1))
    if env_bands != int(n_bands):
        raise ValueError(f"model expects {env_bands} embedding bands but the cube has {n_bands}")
    env_backend = envelope.get("backend_id")
    if env_backend and backend_id and env_backend != backend_id:
        raise ValueError(
            f"model was trained on backend {env_backend!r}, not the current {backend_id!r}"
        )
