"""Shared, torch-free test helpers for the pure-core suite.

Nothing here imports GDAL, QGIS, torch or scikit-learn; the whole ``eocore``
layer is exercised with NumPy alone, so ``pytest`` runs green in a bare sandbox.
GDAL/QGIS-dependent behaviour is covered by ``test_gdal_gated.py``, which skips
cleanly when those libraries are absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_cube(n_bands: int, height: int, width: int, *, seed: int = 0) -> NDArray[Any]:
    """A deterministic random ``(bands, H, W)`` float32 cube for tests."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_bands, height, width)).astype(np.float32)


def unit_cube(n_bands: int, height: int, width: int, *, seed: int = 0) -> NDArray[Any]:
    """Like :func:`make_cube` but L2-normalised along the band axis."""
    cube = make_cube(n_bands, height, width, seed=seed)
    norms = np.sqrt(np.sum(cube * cube, axis=0, keepdims=True))
    normed: NDArray[Any] = (cube / norms).astype(np.float32)
    return normed
