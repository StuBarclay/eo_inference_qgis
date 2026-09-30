"""Nodata-aware temporal compositing of a stack of scenes.

A single date is often partly cloudy, so the fetch step can optionally pull
several scenes over a window and reduce them to one clean cube. All the maths is
pure NumPy and nodata-aware, so it is unit-testable without GDAL: the caller
warps each scene onto the same grid and hands the aligned cubes here.

Compositing is **opt-in** — the default fetch path still returns the single
least-cloudy scene; a composite is built only when the caller asks for one.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float

#: Supported reduction modes (see :func:`composite`).
COMPOSITE_MODES: tuple[str, ...] = ("median", "mean", "first_valid")


def _stack(cubes: Sequence[NDArray[Any]]) -> NDArray[Any]:
    if not cubes:
        raise ValueError("need at least one scene to composite")
    arrays = [as_float(c) for c in cubes]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all scenes must share the same (bands, H, W) shape")
    if arrays[0].ndim != 3:
        raise ValueError(f"expected (bands, H, W) scenes, got shape {shape}")
    return np.stack(arrays, axis=0)


def _masked(stack: NDArray[Any], nodata: float) -> NDArray[Any]:
    """Copy of ``stack`` with nodata / non-finite samples set to NaN."""
    out = stack.astype(np.float64, copy=True)
    invalid = ~np.isfinite(out)
    if not np.isnan(nodata):
        invalid |= out == nodata
    out[invalid] = np.nan
    return out


def composite(
    cubes: Sequence[NDArray[Any]], *, nodata: float = 0.0, mode: str = "median"
) -> NDArray[Any]:
    """Reduce a stack of aligned ``(bands, H, W)`` scenes to one cube.

    ``median``/``mean`` ignore nodata per band per pixel; ``first_valid`` takes,
    for each band and pixel, the value from the earliest scene that is valid
    there. Pixels with no valid observation across the whole stack are filled
    with ``nodata``.
    """
    if mode not in COMPOSITE_MODES:
        raise ValueError(f"unknown composite mode {mode!r}; expected one of {COMPOSITE_MODES}")
    stack = _stack(cubes)
    if mode == "first_valid":
        return _first_valid(stack, nodata)
    masked = _masked(stack, nodata)
    all_nan = np.all(np.isnan(masked), axis=0)
    # All-NaN pixels (fully cloudy/nodata everywhere) are expected and filled
    # below; NumPy still warns on them, so silence that one benign warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reduced = np.nanmedian(masked, axis=0) if mode == "median" else np.nanmean(masked, axis=0)
    reduced = np.where(all_nan, nodata, reduced)
    out: NDArray[Any] = reduced.astype(np.float32, copy=False)
    return out


def _first_valid(stack: NDArray[Any], nodata: float) -> NDArray[Any]:
    n_scenes = stack.shape[0]
    valid = np.isfinite(stack)
    if not np.isnan(nodata):
        valid &= stack != nodata
    # Earliest valid scene index per (band, H, W); n_scenes means "none valid".
    order = np.where(valid, np.arange(n_scenes)[:, None, None, None], n_scenes)
    first = order.min(axis=0)
    none_valid = first == n_scenes
    pick = np.clip(first, 0, n_scenes - 1)
    chosen = np.take_along_axis(stack, pick[None, ...], axis=0)[0]
    out: NDArray[Any] = np.where(none_valid, nodata, chosen).astype(np.float32, copy=False)
    return out
