"""Turn a high-dimensional embedding cube into a 3-band RGB preview.

Two methods, both pure NumPy and unit-testable:

* ``pca`` — fit a 3-component PCA on a subsample of the valid pixels (SVD on the
  mean-centred sample), project every valid pixel onto those components, then
  stretch each of the three channels to 8-bit by a percentile clip.
* ``bands`` — pick three embedding bands directly and percentile-stretch each.

Invalid pixels (nodata / non-finite) are written as black ``(0, 0, 0)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float
from .descriptor import DEFAULT_NODATA
from .ml import valid_mask_from_nodata

METHODS: tuple[str, ...] = ("pca", "bands")


def stretch_values(
    values: NDArray[Any], *, low_pct: float = 2.0, high_pct: float = 98.0
) -> NDArray[Any]:
    """Percentile-clip a 1-D array and scale it to ``uint8`` ``[0, 255]``.

    A constant (or near-constant) channel maps to all zeros rather than dividing
    by zero.
    """
    arr = as_float(values).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out: NDArray[Any] = np.round(scaled * 255.0).astype(np.uint8)
    return out


def fit_pca(
    matrix: NDArray[Any],
    n_components: int = 3,
    *,
    seed: int = 0,
    max_samples: int = 100_000,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Fit a PCA on an ``(n_samples, n_bands)`` matrix.

    Returns ``(mean, components)`` where ``components`` is ``(k, n_bands)`` with
    ``k = min(n_components, n_bands)``. Project new data as
    ``(x - mean) @ components.T``. A deterministic subsample bounds the SVD cost.
    """
    arr = as_float(matrix)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (n_samples, n_bands) matrix, got {arr.shape}")
    n_samples = arr.shape[0]
    if n_samples == 0:
        raise ValueError("need at least one sample for PCA")
    if n_samples > max_samples:
        rng = np.random.default_rng(seed)
        index = rng.choice(n_samples, size=max_samples, replace=False)
        sample = arr[index]
    else:
        sample = arr
    # A NaN/Inf in the fit sample silently poisons the mean and makes the SVD
    # emit non-finite components (or raise LinAlgError), which then paints the
    # whole RGB preview black. Callers pass masked valid pixels, so reaching
    # here with non-finite values is a bug worth surfacing loudly.
    if not np.all(np.isfinite(sample)):
        raise ValueError(
            "PCA input contains non-finite values (NaN/Inf); "
            "mask nodata / invalid pixels out before fitting"
        )
    mean = sample.mean(axis=0)
    centered = sample - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    components: NDArray[Any] = vt[:k]
    return mean, components


def _resolve_triplet(bands: Sequence[int] | None, n_bands: int) -> tuple[int, int, int]:
    if bands is None:
        return (0, min(1, n_bands - 1), min(2, n_bands - 1))
    indices = [int(b) for b in bands]
    if len(indices) != 3:
        raise ValueError("band-triplet mode needs exactly three band indices")
    for index in indices:
        if not 0 <= index < n_bands:
            raise ValueError(f"band index {index} out of range [0, {n_bands})")
    return (indices[0], indices[1], indices[2])


@dataclass(frozen=True)
class RgbTransform:
    """A fitted embedding→RGB mapping that can be applied to any cube (or tile).

    Splitting the fit from the application is what lets the low-memory preview
    path fit the PCA / percentile clip on a bounded pixel sample and then apply
    it tile-by-tile, while the default whole-cube path fits and applies over the
    same pixels for a byte-identical result. ``lo``/``hi`` are the per-channel
    percentile clip bounds; ``mean``/``components`` are set for ``pca`` and
    ``triplet`` for ``bands``.
    """

    method: str
    lo: NDArray[Any]
    hi: NDArray[Any]
    mean: NDArray[Any] | None = None
    components: NDArray[Any] | None = None
    triplet: tuple[int, int, int] | None = None


def _channel_values(
    method: str,
    valid: NDArray[Any],
    *,
    mean: NDArray[Any] | None,
    components: NDArray[Any] | None,
    triplet: tuple[int, int, int] | None,
) -> list[NDArray[Any]]:
    """The three pre-stretch channel vectors for an ``(n, n_bands)`` valid matrix."""
    if method == "pca":
        assert mean is not None and components is not None
        projected = (valid - mean) @ components.T
        available = projected.shape[1]
        return [
            projected[:, c]
            if c < available
            else np.zeros(projected.shape[0], dtype=projected.dtype)
            for c in range(3)
        ]
    assert triplet is not None
    return [valid[:, triplet[c]] for c in range(3)]


def _apply_stretch(values: NDArray[Any], lo: float, hi: float) -> NDArray[Any]:
    """Percentile clip with *fixed* bounds — the apply-time twin of stretch_values."""
    arr = as_float(values).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out: NDArray[Any] = np.round(scaled * 255.0).astype(np.uint8)
    return out


def fit_rgb_transform(
    valid: NDArray[Any],
    *,
    method: str = "pca",
    bands: Sequence[int] | None = None,
    n_bands: int,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    seed: int = 0,
    max_samples: int = 100_000,
) -> RgbTransform:
    """Fit the RGB mapping from an ``(n_samples, n_bands)`` matrix of valid pixels.

    ``n_bands`` is the cube's band count (needed to resolve/validate the band
    triplet even when ``valid`` is a subsample). The percentile clip bounds are
    taken from ``valid``; pass the whole valid set for the exact preview or a
    bounded sample for the low-memory path.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if valid.ndim != 2:
        raise ValueError(f"expected a 2-D (n_samples, n_bands) matrix, got {valid.shape}")
    mean: NDArray[Any] | None = None
    components: NDArray[Any] | None = None
    triplet: tuple[int, int, int] | None = None
    if method == "pca":
        mean, components = fit_pca(valid, 3, seed=seed, max_samples=max_samples)
    else:
        triplet = _resolve_triplet(bands, n_bands)
    channel_vals = _channel_values(method, valid, mean=mean, components=components, triplet=triplet)
    lo = np.array(
        [float(np.percentile(v, low_pct)) if v.size else 0.0 for v in channel_vals],
        dtype=np.float64,
    )
    hi = np.array(
        [float(np.percentile(v, high_pct)) if v.size else 0.0 for v in channel_vals],
        dtype=np.float64,
    )
    return RgbTransform(
        method=method, lo=lo, hi=hi, mean=mean, components=components, triplet=triplet
    )


def apply_rgb_transform(
    cube: NDArray[Any], transform: RgbTransform, *, nodata: float = DEFAULT_NODATA
) -> NDArray[Any]:
    """Apply a fitted :class:`RgbTransform` to a ``(bands, H, W)`` cube (or tile)."""
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (bands, H, W) cube, got shape {arr.shape}")
    n_bands, height, width = arr.shape
    mask = valid_mask_from_nodata(arr, nodata)
    flat_mask = mask.reshape(-1)
    out = np.zeros((3, height, width), dtype=np.uint8)
    if not np.any(flat_mask):
        return out
    valid = arr.reshape(n_bands, -1).T[flat_mask]
    channel_vals = _channel_values(
        transform.method,
        valid,
        mean=transform.mean,
        components=transform.components,
        triplet=transform.triplet,
    )
    for channel in range(3):
        stretched = _apply_stretch(
            channel_vals[channel], transform.lo[channel], transform.hi[channel]
        )
        band = np.zeros(flat_mask.shape[0], dtype=np.uint8)
        band[flat_mask] = stretched
        out[channel] = band.reshape(height, width)
    return out


def embedding_to_rgb(
    cube: NDArray[Any],
    *,
    method: str = "pca",
    bands: Sequence[int] | None = None,
    nodata: float = DEFAULT_NODATA,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    seed: int = 0,
    max_samples: int = 100_000,
) -> NDArray[Any]:
    """Render a ``(bands, H, W)`` embedding cube to a ``(3, H, W)`` uint8 image.

    Convenience wrapper that fits the transform on the cube's own valid pixels and
    applies it — kept for callers and tests that render a whole in-memory cube.
    """
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (bands, H, W) cube, got shape {arr.shape}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    n_bands, height, width = arr.shape
    mask = valid_mask_from_nodata(arr, nodata)
    flat_mask = mask.reshape(-1)
    if not np.any(flat_mask):
        return np.zeros((3, height, width), dtype=np.uint8)
    valid = arr.reshape(n_bands, -1).T[flat_mask]
    transform = fit_rgb_transform(
        valid,
        method=method,
        bands=bands,
        n_bands=n_bands,
        low_pct=low_pct,
        high_pct=high_pct,
        seed=seed,
        max_samples=max_samples,
    )
    return apply_rgb_transform(arr, transform, nodata=nodata)
