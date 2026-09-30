"""Torch-free tests for the shared optional-dependency helpers (M8).

``eocore.backends._deps`` is the single source of truth for detecting optional
model packages and running the opt-in ``pip install`` that the sklearn tier, the
STAC signer and the GUI all delegate to — but it was previously untested. Every
function here is exercised without importing torch or actually installing
anything: package detection is driven through real stdlib / known-absent module
names (and a monkeypatched ``find_spec`` for the defensive error arm), and
``pip_install`` is checked against a monkeypatched ``subprocess.run`` so the exact
command it builds is asserted without spawning pip.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from typing import Any

import pytest

from eo_inference_toolbox.eocore.backends import _deps
from eo_inference_toolbox.eocore.backends._deps import (
    TORCH_INDEX_URLS,
    _quote,
    install_hint,
    missing_import_names,
    missing_packages,
    module_available,
    optional_import_error,
    pip_install,
    torch_available,
    torch_index_url,
)
from eo_inference_toolbox.eocore.backends.base import backend_info

# -- torch wheel index selection ---------------------------------------------


def test_torch_index_url_known_variants() -> None:
    assert torch_index_url("default") is None
    assert torch_index_url("cpu") == "https://download.pytorch.org/whl/cpu"
    assert torch_index_url("cuda") == "https://download.pytorch.org/whl/cu121"


def test_torch_index_url_unknown_variant_is_none() -> None:
    # .get() fallback: an unrecognised variant must not raise, just yield None.
    assert torch_index_url("nonsense") is None


def test_torch_index_table_covers_the_documented_variants() -> None:
    assert set(TORCH_INDEX_URLS) == {"default", "cpu", "cuda"}


# -- module_available ---------------------------------------------------------


def test_module_available_true_for_stdlib() -> None:
    assert module_available("json") is True


def test_module_available_false_for_absent_top_level() -> None:
    assert module_available("definitely_absent_module_xyz") is False


def test_module_available_false_when_parent_is_not_a_package() -> None:
    # find_spec raises ModuleNotFoundError (an ImportError) here; it must be
    # swallowed and reported as simply "not available".
    assert module_available("os.path.nonexistent_submodule") is False


def test_module_available_swallows_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # find_spec raises ValueError for some malformed specs; the helper's
    # (ImportError, ValueError) guard must turn that into False, not propagate.
    def boom(_name: str) -> Any:
        raise ValueError("malformed spec")

    # _deps did ``import importlib.util``, so this is the same module object it
    # calls find_spec on; patching here steers _deps.module_available.
    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert module_available("whatever") is False


def test_torch_available_delegates_to_module_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_deps, "module_available", lambda name: name == "torch")
    assert torch_available() is True
    monkeypatch.setattr(_deps, "module_available", lambda _name: False)
    assert torch_available() is False


# -- missing_import_names / missing_packages ----------------------------------


def test_missing_import_names_prepends_torch_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A backend that already lists torch among its import names must not report it
    # twice, and torch is always considered required.
    info = dataclasses.replace(backend_info("clay"), import_names=("torch", "foo"))
    monkeypatch.setattr(_deps, "module_available", lambda _name: False)
    assert missing_import_names(info) == ["torch", "foo"]


def test_missing_import_names_filters_to_absent_only(monkeypatch: pytest.MonkeyPatch) -> None:
    info = dataclasses.replace(backend_info("clay"), import_names=("foo", "bar"))
    monkeypatch.setattr(_deps, "module_available", lambda name: name in {"torch", "foo"})
    # torch + foo importable, bar absent -> only bar is missing.
    assert missing_import_names(info) == ["bar"]


def test_missing_packages_empty_when_all_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_deps, "module_available", lambda _name: True)
    assert missing_packages("clay") == []


def test_missing_packages_returns_pip_specs_when_unsatisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_deps, "module_available", lambda _name: False)
    assert missing_packages("clay") == list(backend_info("clay").pip_packages)


# -- install_hint / _quote ----------------------------------------------------


def test_quote_only_wraps_specs_with_shell_significant_chars() -> None:
    assert _quote("torch") == "torch"
    assert _quote("satlaspretrain-models") == "satlaspretrain-models"
    assert _quote("torch>=2.0") == '"torch>=2.0"'
    assert _quote("package<3") == '"package<3"'
    assert _quote("a b") == '"a b"'
    assert _quote("pkg;extra") == '"pkg;extra"'


def test_install_hint_builds_pip_command_for_backend() -> None:
    hint = install_hint("clay")
    assert hint.startswith(f'"{sys.executable}" -m pip install ')
    # Every declared spec appears; plain specs are unquoted (no shell-significant chars).
    for spec in backend_info("clay").pip_packages:
        assert spec in hint


# -- pip_install --------------------------------------------------------------


class _FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_pip_install_noop_for_empty_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> _FakeCompleted:
        calls.append(cmd)
        return _FakeCompleted(0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, message = pip_install([])
    assert ok is True
    assert message == "nothing to install"
    assert calls == []  # subprocess is never spawned for an empty request


def test_pip_install_success_builds_command_with_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> _FakeCompleted:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeCompleted(0, "installed ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, output = pip_install(
        ["somepkg"], upgrade=True, timeout=123, index_url="https://example/whl"
    )
    assert ok is True
    assert output == "installed ok\n"
    assert captured["cmd"] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--index-url",
        "https://example/whl",
        "somepkg",
    ]
    # Consent-gated, bounded, and captured — never inherits the parent's streams.
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["timeout"] == 123


def test_pip_install_defaults_omit_upgrade_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **_kwargs: Any) -> _FakeCompleted:
        captured["cmd"] = cmd
        return _FakeCompleted(0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pip_install(["pkg"])
    assert captured["cmd"] == [sys.executable, "-m", "pip", "install", "pkg"]


def test_pip_install_failure_reports_combined_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(_cmd: list[str], **_kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(1, "stdout-part\n", "stderr-part\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, output = pip_install(["pkg"])
    assert ok is False
    # stdout and stderr are concatenated so the caller sees the whole failure.
    assert output == "stdout-part\nstderr-part\n"


# -- optional_import_error ----------------------------------------------------


def test_optional_import_error_absent_package_says_install() -> None:
    # A genuinely missing top-level module: module_available() is False, so the
    # message tells the user to install it, and still surfaces the real detail.
    msg = optional_import_error(
        ImportError("No module named 'definitely_absent_xyz'"),
        label="TerraTorch",
        module="definitely_absent_module_xyz",
        install_cmd="pip install terratorch",
    )
    assert "Could not import TerraTorch" in msg
    assert "Install it with: pip install terratorch" in msg
    assert "definitely_absent_xyz" in msg  # underlying detail is surfaced
    assert "installed but failed to import" not in msg  # not misreported


def test_optional_import_error_installed_but_broken_is_distinguished() -> None:
    # "os" is always importable, so module_available() is True: this stands in for
    # a package present on disk whose deeper import raised (a broken dependency).
    msg = optional_import_error(
        ImportError("cannot import name 'BACKBONE_REGISTRY'"),
        label="TerraTorch",
        module="os",
        install_cmd="pip install terratorch",
    )
    assert "is installed but failed to import" in msg
    assert "will not help" in msg  # reinstalling is not the fix
    assert "cannot import name 'BACKBONE_REGISTRY'" in msg  # real cause shown
    assert "Could not import TerraTorch" not in msg  # not the absent-package text


def test_optional_import_error_includes_install_cmd_and_docs() -> None:
    msg = optional_import_error(
        ImportError("boom"),
        label="X",
        module="definitely_absent_module_xyz",
        install_cmd="pip install x",
        docs="docs/CUSTOM.md",
    )
    assert "pip install x" in msg
    assert "docs/CUSTOM.md" in msg


def test_optional_import_error_empty_detail_falls_back_to_classname() -> None:
    class WeirdImportError(ImportError):
        pass

    # str(exc) is empty -> the exception class name stands in, so there is never a
    # dangling "(import error: )".
    msg = optional_import_error(
        WeirdImportError(),
        label="X",
        module="definitely_absent_module_xyz",
        install_cmd="pip install x",
    )
    assert "WeirdImportError" in msg


def test_optional_import_error_branches_on_module_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Drive the branch purely through module_available so the test needs no
    # particular package to be installed or absent.
    monkeypatch.setattr(_deps, "module_available", lambda _name: True)
    broken = optional_import_error(
        ImportError("x"), label="Pkg", module="whatever", install_cmd="pip install pkg"
    )
    assert "installed but failed to import" in broken

    monkeypatch.setattr(_deps, "module_available", lambda _name: False)
    absent = optional_import_error(
        ImportError("x"), label="Pkg", module="whatever", install_cmd="pip install pkg"
    )
    assert "Could not import Pkg" in absent


def test_optional_import_error_windows_dll_clash_adds_native_hint() -> None:
    # A real Windows native-extension failure: the package is present ("os" stands
    # in for an installed-but-broken package) but a compiled dependency bound to the
    # wrong build of a shared native library. The message keeps the installed-but-
    # broken framing AND appends the native-library-clash hint.
    msg = optional_import_error(
        ImportError(
            "DLL load failed while importing _errors: The specified procedure could not be found."
        ),
        label="TerraTorch",
        module="os",
        install_cmd="pip install terratorch",
    )
    assert "is installed but failed to import" in msg  # base framing kept
    assert "native-library clash" in msg  # targeted hint added
    assert "one DLL of a given base name" in msg  # explains the Windows mechanism
    assert "matches the version QGIS ships" in msg  # points at the fix


def test_optional_import_error_broken_but_not_dll_has_no_native_hint() -> None:
    # An installed-but-broken import whose cause is a plain Python ImportError, not a
    # native DLL-load failure, must NOT get the native-library-clash hint.
    msg = optional_import_error(
        ImportError("cannot import name 'BACKBONE_REGISTRY'"),
        label="TerraTorch",
        module="os",
        install_cmd="pip install terratorch",
    )
    assert "is installed but failed to import" in msg
    assert "native-library clash" not in msg


def test_optional_import_error_absent_package_ignores_dll_wording() -> None:
    # DLL-load wording on a genuinely absent package still routes to the "install it"
    # path (the native-clash hint only applies to installed-but-broken imports).
    msg = optional_import_error(
        ImportError("DLL load failed while importing _errors"),
        label="TerraTorch",
        module="definitely_absent_module_xyz",
        install_cmd="pip install terratorch",
    )
    assert "Could not import TerraTorch" in msg
    assert "Install it with: pip install terratorch" in msg
    assert "native-library clash" not in msg


def test_looks_like_native_library_clash_matches_windows_signatures() -> None:
    # Error 127 (wrong-version DLL, missing symbol), error 126 (dependent DLL absent),
    # and the generic prefix — all matched, case-insensitively.
    assert _deps._looks_like_native_library_clash(
        "DLL load failed while importing _errors: The specified procedure could not be found."
    )
    assert _deps._looks_like_native_library_clash(
        "ImportError: The specified module could not be found."
    )
    assert _deps._looks_like_native_library_clash("dll LOAD failed")  # case-insensitive
    # A non-DLL Python import error is not a native-library clash.
    assert not _deps._looks_like_native_library_clash(
        "cannot import name 'BACKBONE_REGISTRY' from 'terratorch.registry'"
    )
