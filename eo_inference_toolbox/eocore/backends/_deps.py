"""Detect optional model dependencies and offer a guided install.

This module is the **single source of truth** for two things the rest of the
plugin used to re-implement: checking whether an optional package is importable
(:func:`module_available`) and running an opt-in ``pip install``
(:func:`pip_install`). The scikit-learn tier, the STAC signer and the GUI all
delegate here so the behaviour and policy live in one place.

Nothing heavy is imported at module load; callers ask :func:`missing_packages`
what is absent and either surface a clear message or (only on explicit opt-in)
call :func:`pip_install`. Weights are never handled here — see :mod:`._weights`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Sequence

from .base import BackendInfo, backend_info

# PyTorch wheel index URLs, kept in one place so they are easy to update as the
# upstream CUDA build tags move. ``default`` uses PyPI (whatever wheel pip picks
# for the platform); ``cpu`` forces the CPU-only build; ``cuda`` targets a CUDA
# build. Callers pass one of these to :func:`pip_install` as ``index_url``.
TORCH_INDEX_URLS: dict[str, str | None] = {
    "default": None,
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu121",
}


def torch_index_url(variant: str = "default") -> str | None:
    """Return the pip ``--index-url`` for a torch build variant (or ``None``)."""
    return TORCH_INDEX_URLS.get(variant)


def module_available(import_name: str) -> bool:
    """True if ``import_name`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def torch_available() -> bool:
    """True if PyTorch is importable."""
    return module_available("torch")


# Substrings (matched case-insensitively) that mark a Windows native-extension
# DLL-load failure. "specified procedure could not be found" is error 127 — a DLL
# loaded but is the wrong version, missing a symbol; "specified module could not be
# found" is error 126 — a dependent DLL is absent; "dll load failed" is the generic
# prefix Python prepends to both.
_WINDOWS_DLL_LOAD_MARKERS = (
    "dll load failed",
    "specified procedure could not be found",
    "specified module could not be found",
)


def _looks_like_native_library_clash(detail: str) -> bool:
    """True if ``detail`` reads like a Windows native-extension DLL-load failure.

    These are raised when a compiled extension binds to the wrong build of a shared
    native library — classically because QGIS has already loaded its own copy of that
    library (HDF5, GDAL, …) and Windows permits only one DLL of a given base name per
    process, so the extension gets QGIS's copy instead of the one it shipped with.
    """
    lowered = detail.lower()
    return any(marker in lowered for marker in _WINDOWS_DLL_LOAD_MARKERS)


def optional_import_error(
    exc: ImportError,
    *,
    label: str,
    module: str,
    install_cmd: str,
    docs: str = "docs/SMOKE_TEST.md",
) -> str:
    """Build an actionable message for a failed optional-package import.

    A backend catches the ``ImportError`` from importing its model package and
    passes it here. The message distinguishes two very different situations that
    the old fixed "install it" text conflated:

    * the package is genuinely **absent** (``module`` is not even on the import
      path) — ``pip install`` is the fix; and
    * the package is **installed but fails to import** — almost always a broken
      or incompatible dependency in the interpreter, where reinstalling the
      top-level package changes nothing.

    :func:`module_available` uses :func:`importlib.util.find_spec`, which locates
    a package without executing it, so it stays true for an installed-but-broken
    package even though importing it raised. Either way the underlying import
    error is surfaced rather than hidden. ``module`` must be the *top-level*
    importable name (e.g. ``"terratorch"``), never a submodule.

    When the installed-but-broken import carries a Windows native-DLL-load
    signature, a further hint names the likely cause — a native-library version
    clash with QGIS — because that is not something reinstalling the Python
    package can fix (see the worked example in the User Guide troubleshooting).
    """
    detail = str(exc).strip() or exc.__class__.__name__
    if module_available(module):
        message = (
            f"{label} is installed but failed to import, which usually means a "
            f"broken or incompatible dependency in the QGIS Python rather than a "
            f"missing package (underlying import error: {detail}). Reinstalling "
            f"with '{install_cmd}' will not help until that is resolved — run the "
            f"import in the QGIS Python Console to see the full traceback, and see "
            f"{docs}."
        )
        if _looks_like_native_library_clash(detail):
            message += (
                " This particular error looks like a Windows native-library clash: "
                "a compiled extension is binding to a different build of a shared "
                "native library (such as HDF5 or GDAL) than the one QGIS already "
                "loaded, because Windows permits only one DLL of a given base name "
                "per process. The usual fix is to pin the offending package to a "
                "build whose bundled native library matches the version QGIS ships."
            )
        return message
    return (
        f"Could not import {label} (import error: {detail}). Install it with: "
        f"{install_cmd}, then confirm the import in the QGIS Python Console "
        f"(see {docs})."
    )


def missing_import_names(info: BackendInfo) -> list[str]:
    """Return the import names a backend needs that are not importable."""
    names = ("torch", *info.import_names)
    missing: list[str] = []
    for name in names:
        if not module_available(name) and name not in missing:
            missing.append(name)
    return missing


def missing_packages(backend_id: str) -> list[str]:
    """Return the pip specs to install for a backend, or ``[]`` if satisfied."""
    info = backend_info(backend_id)
    if not missing_import_names(info):
        return []
    return list(info.pip_packages)


def install_hint(backend_id: str) -> str:
    """A copy-pasteable pip command for a backend's dependencies."""
    specs = backend_info(backend_id).pip_packages
    quoted = " ".join(_quote(spec) for spec in specs)
    return f'"{sys.executable}" -m pip install {quoted}'


def _quote(spec: str) -> str:
    return f'"{spec}"' if any(ch in spec for ch in " <>=;") else spec


def pip_install(
    packages: Sequence[str],
    *,
    upgrade: bool = False,
    timeout: int = 3600,
    index_url: str | None = None,
) -> tuple[bool, str]:
    """Install ``packages`` into the running interpreter. Opt-in only.

    Returns ``(ok, combined_output)``. The caller must have obtained explicit
    user consent — this never runs implicitly. ``index_url`` (e.g. a PyTorch
    CPU/CUDA wheel index) is passed through as ``--index-url`` when given.
    """
    if not packages:
        return True, "nothing to install"
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(packages)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return False, f"pip invocation failed: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output
