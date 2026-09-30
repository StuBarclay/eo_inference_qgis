"""Guard against version drift across the package, metadata.txt and pyproject.

QGIS shows the ``metadata.txt`` version in its plugin manager; packaging tools use
``pyproject.toml``; code that reports its own version reads ``__version__``. These
three must agree, or a release ships mislabelled. The package ``__init__`` imports
nothing from QGIS at module level, so this stays a pure, sandbox-runnable test.
"""

from __future__ import annotations

import re
from pathlib import Path

import eo_inference_toolbox


def _pkg_dir() -> Path:
    init = eo_inference_toolbox.__file__
    assert init is not None  # a regular (non-namespace) package always has __file__
    return Path(init).parent


def _metadata_version() -> str:
    text = (_pkg_dir() / "metadata.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("version="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("metadata.txt has no 'version=' line")


def _pyproject_version() -> str:
    text = (_pkg_dir().parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "pyproject.toml has no 'version = \"...\"' line"
    return match.group(1)


def test_package_version_matches_metadata() -> None:
    assert eo_inference_toolbox.__version__ == _metadata_version()


def test_package_version_matches_pyproject() -> None:
    assert eo_inference_toolbox.__version__ == _pyproject_version()
