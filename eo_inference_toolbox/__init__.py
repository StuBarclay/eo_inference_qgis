"""EO Inference Toolbox — a QGIS Processing plugin.

Run Earth-observation foundation models — Clay v1.5, Prithvi-EO-2.0 and
SatlasPretrain — over your *own* imagery to produce per-pixel embedding rasters,
then explore them with a self-contained similarity / change / classification
workbench.

None of the three models ships a global pre-computed embedding archive: each is
an *inference backend* that you run over source imagery (Sentinel-2, HLS, …).
The heavy machinery (PyTorch, downloaded weights, imagery acquisition) is an
opt-in tier that is detected at run time and never a hard dependency, so this
module stays deliberately import-light: it only wires the QGIS entry point.
"""

from __future__ import annotations

import sys
from typing import Any

__version__ = "0.2.0"

_MIN_PYTHON = (3, 10)


def classFactory(iface: Any) -> Any:
    """Return the plugin instance. Called by QGIS at start-up."""
    if sys.version_info < _MIN_PYTHON:
        running = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "EO Inference Toolbox requires Python "
            f"{_MIN_PYTHON[0]}.{_MIN_PYTHON[1]} or newer (running {running})."
        )
    from .plugin import EoInferencePlugin

    return EoInferencePlugin(iface)
