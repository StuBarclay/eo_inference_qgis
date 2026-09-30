"""Change and trajectory metrics between embedding cubes, descriptor-aware."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float
from .normalize import l2_normalize

METRICS: tuple[str, ...] = ("cosine", "euclidean")


def distance_map(
    a: NDArray[Any],
    b: NDArray[Any],
    *,
    metric: str = "cosine",
    assume_unit: bool = False,
    eps: float = 1e-12,
) -> NDArray[Any]:
    """Per-pixel distance between two ``(bands, H, W)`` cubes.

    ``cosine`` returns ``1 - cos`` in ``[0, 2]``; ``euclidean`` returns the L2
    distance. For cosine on non-unit inputs the cubes are L2-normalised first.
    """
    arr_a = as_float(a)
    arr_b = as_float(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"cube shapes differ: {arr_a.shape} vs {arr_b.shape}")
    if metric == "cosine":
        if not assume_unit:
            arr_a = l2_normalize(arr_a, eps=eps)
            arr_b = l2_normalize(arr_b, eps=eps)
        dot = np.sum(arr_a * arr_b, axis=0)
        # Clamp tiny floating-point overshoot so cosine distance is a hard
        # [0, 2] as documented (sqrt/log/colour ramps downstream rely on it).
        dist = np.clip(1.0 - dot, 0.0, 2.0)
    elif metric == "euclidean":
        diff = arr_a - arr_b
        dist = np.sqrt(np.sum(diff * diff, axis=0))
    else:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    out: NDArray[Any] = np.asarray(dist).astype(arr_a.dtype, copy=False)
    return out


def mean_embedding(cubes: Sequence[NDArray[Any]], *, nodata: float | None = None) -> NDArray[Any]:
    """Element-wise mean over a sequence of same-shaped ``(bands, H, W)`` cubes.

    With ``nodata=None`` (the default) this is a plain element-wise mean: every
    cube contributes to every pixel, byte-for-byte as before. The change
    algorithm's anomaly baseline uses this default because it already intersects
    a per-cube :func:`~eo_inference_toolbox.eocore.ml.valid_mask_from_nodata`
    over *all* cubes and writes output only where every cube is valid, so no
    sentinel can reach a written pixel.

    When ``nodata`` is given, a pixel is treated as *missing* in a cube when
    every band there equals the sentinel (the "all bands" rule shared with
    :func:`~eo_inference_toolbox.eocore.ml.valid_mask_from_nodata`), and each
    pixel is averaged over only the cubes in which it is present; a pixel that is
    missing in *every* cube is written as ``nodata``. This keeps a ``-9999`` fill
    out of the average for callers that mean cubes directly, without a combining
    mask. Non-finite values are not treated as missing here — pass finite cubes,
    or intersect a validity mask, if that matters.
    """
    if not cubes:
        raise ValueError("need at least one cube")
    arrays = [as_float(c) for c in cubes]
    stack = np.stack(arrays, axis=0)
    if nodata is None or not np.isfinite(nodata):
        plain: NDArray[Any] = np.mean(stack, axis=0)
        return plain
    sentinel = np.float32(nodata)
    # (N, 1, H, W): True where a cube is the all-band fill at that pixel.
    missing = np.all(stack == sentinel, axis=1, keepdims=True)
    present = ~missing
    present_count = present.sum(axis=0)  # (1, H, W): how many cubes cover each pixel
    contributions = np.where(present, stack, np.float32(0.0))
    summed = contributions.sum(axis=0)  # (bands, H, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = summed / present_count  # (1, H, W) broadcasts over the band axis
    averaged[:, present_count[0] == 0] = sentinel  # all cubes missing -> nodata
    out: NDArray[Any] = averaged.astype(arrays[0].dtype, copy=False)
    return out


def trajectory_magnitude(
    cubes: Sequence[NDArray[Any]],
    *,
    metric: str = "cosine",
    assume_unit: bool = False,
    eps: float = 1e-12,
) -> NDArray[Any]:
    """Sum of consecutive pairwise distances along an ordered series of cubes."""
    arrays = [as_float(c) for c in cubes]
    if len(arrays) < 2:
        raise ValueError("need at least two cubes for a trajectory")
    pairs = list(itertools.pairwise(arrays))
    (first_a, first_b) = pairs[0]
    result = distance_map(first_a, first_b, metric=metric, assume_unit=assume_unit, eps=eps)
    for prev, curr in pairs[1:]:
        result = result + distance_map(prev, curr, metric=metric, assume_unit=assume_unit, eps=eps)
    out: NDArray[Any] = result
    return out


def anomaly_from_baseline(
    cube: NDArray[Any],
    baseline: NDArray[Any],
    *,
    metric: str = "cosine",
    assume_unit: bool = False,
    eps: float = 1e-12,
) -> NDArray[Any]:
    """Distance of a cube from a baseline embedding (e.g. a temporal mean)."""
    return distance_map(cube, baseline, metric=metric, assume_unit=assume_unit, eps=eps)
