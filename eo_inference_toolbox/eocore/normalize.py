"""L2 normalisation — the correctness linchpin of the toolbox.

The similarity and change metrics take cosine == dot-product shortcuts that are
only valid for unit-length vectors. None of the three backends emits unit
vectors, so every embedding raster this plugin writes is L2-normalised first
(``ensure_unit``), and the metrics stay exactly correct downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float
from .descriptor import EmbeddingDescriptor


def l2_normalize(cube: NDArray[Any], *, eps: float = 1e-12) -> NDArray[Any]:
    """Normalise a ``(bands, rows, cols)`` cube to unit length along the band axis.

    Zero-magnitude pixels are left unchanged (they stay at ~0). Preserves the
    input floating dtype (float32 stays float32).
    """
    arr = as_float(cube)
    norms = np.sqrt(np.sum(arr * arr, axis=0, keepdims=True))
    safe = np.where(norms > eps, norms, 1.0).astype(arr.dtype, copy=False)
    out: NDArray[Any] = arr / safe
    return out


def normalize_rows(matrix: NDArray[Any], *, eps: float = 1e-12) -> NDArray[Any]:
    """Normalise each row of an ``(n_samples, n_bands)`` matrix to unit length."""
    arr = as_float(matrix)
    norms = np.sqrt(np.sum(arr * arr, axis=1, keepdims=True))
    safe = np.where(norms > eps, norms, 1.0).astype(arr.dtype, copy=False)
    out: NDArray[Any] = arr / safe
    return out


def ensure_unit(
    cube: NDArray[Any], descriptor: EmbeddingDescriptor, *, eps: float = 1e-12
) -> NDArray[Any]:
    """Return a unit-normalised cube, honouring the descriptor.

    If the source is already unit-normalised the cube is returned unchanged
    (only coerced to a floating dtype); otherwise it is L2-normalised.
    """
    if descriptor.is_unit_normalized:
        return as_float(cube)
    return l2_normalize(cube, eps=eps)
