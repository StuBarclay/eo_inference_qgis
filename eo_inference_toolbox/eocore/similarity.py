"""Per-pixel cosine similarity, descriptor-aware.

For unit-length embeddings, cosine similarity is a plain dot product. This module
never *assumes* the input is unit-length: unless the caller passes
``assume_unit=True`` it L2-normalises first, so it is correct for the raw,
non-normalised vectors the backends emit as well as for the normalised rasters
this plugin writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float
from .normalize import l2_normalize


def cosine_similarity_map(
    cube: NDArray[Any],
    reference: NDArray[Any],
    *,
    assume_unit: bool = False,
    eps: float = 1e-12,
) -> NDArray[Any]:
    """Cosine similarity of each ``(bands, H, W)`` pixel to a ``(bands,)`` vector.

    Returns a ``(H, W)`` array in ``[-1, 1]``.
    """
    arr = as_float(cube)
    ref: NDArray[Any] = as_float(np.asarray(reference)).reshape(-1)
    if ref.shape[0] != arr.shape[0]:
        raise ValueError(f"reference length {ref.shape[0]} != number of bands {arr.shape[0]}")
    if not assume_unit:
        arr = l2_normalize(arr, eps=eps)
        norm = float(np.sqrt(np.sum(ref * ref)))
        ref = ref / (norm if norm > eps else 1.0)
    sim = np.tensordot(ref, arr, axes=([0], [0]))
    # Clamp tiny floating-point overshoot so the documented [-1, 1] range is a
    # hard guarantee for downstream consumers (fixed colour ramps, acos, ...).
    sim = np.clip(sim, -1.0, 1.0)
    out: NDArray[Any] = sim.astype(arr.dtype, copy=False)
    return out


def reference_from_points(
    cube: NDArray[Any],
    rowcols: Sequence[tuple[int, int]],
    *,
    normalize: bool = True,
    nodata: float | None = None,
    eps: float = 1e-12,
) -> NDArray[Any]:
    """Mean embedding vector over the given (row, col) pixels of a cube.

    Points whose pixel vector is non-finite, or (when ``nodata`` is given) all
    equal to the ``nodata`` sentinel, are dropped before averaging: a reference
    point that lands on an uncovered/void pixel (embedding rasters stamp a
    ``-9999`` collar around every scene) would otherwise poison the mean and
    silently corrupt the whole similarity map. Raises ``ValueError`` if no valid
    point remains.
    """
    arr = as_float(cube)
    if not rowcols:
        raise ValueError("need at least one (row, col) point")
    vectors: list[NDArray[Any]] = []
    for row, col in rowcols:
        vec = arr[:, row, col]
        if not bool(np.all(np.isfinite(vec))):
            continue
        if nodata is not None and not bool(np.any(vec != nodata)):
            continue
        vectors.append(vec)
    if not vectors:
        raise ValueError("no valid reference points: all are non-finite or nodata")
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    if normalize:
        norm = float(np.sqrt(np.sum(mean * mean)))
        mean = mean / (norm if norm > eps else 1.0)
    out: NDArray[Any] = mean.astype(arr.dtype, copy=False)
    return out
