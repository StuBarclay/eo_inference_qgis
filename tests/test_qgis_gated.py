"""QGIS-gated tests for provider / algorithm registration.

These skip automatically when the ``qgis`` Python bindings are absent (the lint
sandbox), and provide a light structural check inside a real QGIS Python.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qgis.core")

from qgis.core import QgsProcessingProvider

from eo_inference_toolbox.algorithms import iter_algorithms
from eo_inference_toolbox.provider import (
    PROVIDER_ID,
    EoInferenceProvider,
)

_EXPECTED_NAMES = {
    "fetchimagery",
    "embed",
    "torgb",
    "similarity",
    "change",
    "classify",
    "cluster",
}


def test_provider_id_is_eoinfer() -> None:
    assert PROVIDER_ID == "eoinfer"
    # Must actually subclass the QGIS provider base, or QGIS will not register it
    # (issubclass against ``object`` would be vacuously true for any class).
    assert issubclass(EoInferenceProvider, QgsProcessingProvider)


def test_iter_algorithms_yields_all_seven() -> None:
    names = {alg.name() for alg in iter_algorithms()}  # type: ignore[attr-defined]
    assert names == _EXPECTED_NAMES
