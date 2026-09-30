"""Tiling and mosaic maths for running a fixed-input model over a large raster.

A ViT backend consumes fixed-size chips, so a scene is cut into overlapping
windows, each run through the model, and the per-pixel outputs stitched back
with overlap averaging. All of this is pure NumPy and unit-testable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float


@dataclass(frozen=True)
class Window:
    """A read window into a raster, in pixel coordinates."""

    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def col_end(self) -> int:
        return self.col_off + self.width

    @property
    def row_end(self) -> int:
        return self.row_off + self.height


def _starts(size: int, tile: int, step: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, step))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_windows(width: int, height: int, tile: int, overlap: int = 0) -> Iterator[Window]:
    """Yield windows covering the whole raster.

    Interior windows are full ``tile`` by ``tile`` chips; the final row/column of
    windows is shifted to sit flush against the edge (so it overlaps its
    neighbour) rather than being clipped, which keeps every chip full-size for a
    fixed-input model. When ``size <= tile`` a single, possibly partial window is
    returned.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile <= 0:
        raise ValueError("tile must be positive")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    step = tile - overlap
    for r in _starts(height, tile, step):
        h = min(tile, height - r)
        for c in _starts(width, tile, step):
            w = min(tile, width - c)
            yield Window(col_off=c, row_off=r, width=w, height=h)


class MosaicAccumulator:
    """Accumulate per-window band outputs into one mosaic, averaging overlaps."""

    def __init__(self, n_bands: int, height: int, width: int, *, dtype: Any = np.float32) -> None:
        if n_bands <= 0 or height <= 0 or width <= 0:
            raise ValueError("n_bands, height and width must be positive")
        self.n_bands = n_bands
        self.height = height
        self.width = width
        self._out_dtype = dtype
        self._sum: NDArray[Any] = np.zeros((n_bands, height, width), dtype=np.float64)
        self._weight: NDArray[Any] = np.zeros((height, width), dtype=np.float64)

    def add(self, window: Window, tile: NDArray[Any], weight: float | NDArray[Any] = 1.0) -> None:
        arr = as_float(tile)
        expected = (self.n_bands, window.height, window.width)
        if arr.shape != expected:
            raise ValueError(f"tile shape {arr.shape} != expected {expected}")
        if isinstance(weight, np.ndarray):
            wexpected = (window.height, window.width)
            if weight.shape != wexpected:
                raise ValueError(f"weight shape {weight.shape} != expected {wexpected}")
            w2d = as_float(weight)
        else:
            w2d = np.full((window.height, window.width), float(weight), dtype=np.float64)
        r0, r1 = window.row_off, window.row_end
        c0, c1 = window.col_off, window.col_end
        self._sum[:, r0:r1, c0:c1] += arr * w2d[np.newaxis, :, :]
        self._weight[r0:r1, c0:c1] += w2d

    @property
    def coverage(self) -> NDArray[Any]:
        """Boolean (rows, cols) mask of pixels that received at least one window."""
        out: NDArray[Any] = self._weight > 0
        return out

    def result(self, *, fill: float = 0.0) -> NDArray[Any]:
        """Return the averaged ``(bands, rows, cols)`` mosaic; unseen pixels = fill."""
        weight = self._weight[np.newaxis, :, :]
        safe = np.where(weight > 0, weight, 1.0)
        averaged = self._sum / safe
        filled = np.where(weight > 0, averaged, fill)
        out: NDArray[Any] = filled.astype(self._out_dtype, copy=False)
        return out


def cosine_taper_weights(height: int, width: int, taper: int) -> NDArray[Any]:
    """A ``(height, width)`` blend weight that fades to (but never reaches) 0 at edges.

    Overlap averaging with uniform weight gives every window an equal vote, so a
    seam is visible wherever two chips disagree. Weighting each chip by a cosine
    taper — full weight in the interior, tapering towards the border — feathers
    those seams. The taper is kept **strictly positive** everywhere: an edge
    pixel covered by only one window must still contribute, otherwise it would be
    flagged as uncovered. ``taper<=0`` returns all-ones (equivalent to uniform).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if taper <= 0:
        return np.ones((height, width), dtype=np.float64)

    def _axis(size: int) -> NDArray[Any]:
        w = np.ones(size, dtype=np.float64)
        span = min(taper, (size + 1) // 2)
        if span <= 0:
            return w
        # Half-cosine ramp in (0, 1]; +1 offsets so the very edge is > 0.
        ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(span) + 1) / (span + 1)))
        w[:span] = ramp
        w[size - span :] = np.minimum(w[size - span :], ramp[::-1])
        return w

    row = _axis(height)
    col = _axis(width)
    out: NDArray[Any] = np.outer(row, col)
    return out
