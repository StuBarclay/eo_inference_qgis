"""Pure-NumPy helpers to turn model outputs into a per-pixel embedding tile.

ViT backends emit a coarse patch/token grid; FPN backends emit several feature
maps at different scales. Both must be resampled back to the tile's pixel grid
so every backend can present the same ``(D, h, w)`` interface. These helpers own
no torch, so they are unit-testable in the sandbox.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._num import as_float

#: Supported patch-grid upsampling methods (see :func:`upsample`).
UPSAMPLING_METHODS: tuple[str, ...] = ("nearest", "bilinear")


def upsample_nearest(grid: NDArray[Any], out_h: int, out_w: int) -> NDArray[Any]:
    """Nearest-neighbour upsample a ``(D, gh, gw)`` grid to ``(D, out_h, out_w)``."""
    arr = as_float(grid)
    if arr.ndim != 3:
        raise ValueError(f"expected (D, gh, gw), got shape {arr.shape}")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output size must be positive")
    _, gh, gw = arr.shape
    if gh == 0 or gw == 0:
        raise ValueError("input grid has a zero spatial dimension")
    rows = (np.arange(out_h) * gh // out_h).clip(0, gh - 1)
    cols = (np.arange(out_w) * gw // out_w).clip(0, gw - 1)
    out: NDArray[Any] = arr[:, rows][:, :, cols]
    return out


def _axis_weights(size_out: int, size_in: int) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Half-pixel-centre bilinear sample indices/weights for one axis.

    Returns ``(idx0, idx1, frac)`` where the interpolated value along the axis is
    ``(1 - frac) * src[idx0] + frac * src[idx1]``. Uses the ``align_corners=False``
    convention so upsampling is consistent regardless of the scale factor.
    """
    if size_in == 1:
        zeros = np.zeros(size_out, dtype=np.intp)
        return zeros, zeros, np.zeros(size_out, dtype=np.float64)
    coord = (np.arange(size_out) + 0.5) * (size_in / size_out) - 0.5
    coord = np.clip(coord, 0.0, size_in - 1)
    idx0 = np.floor(coord).astype(np.intp)
    idx1 = np.minimum(idx0 + 1, size_in - 1)
    frac = coord - idx0
    return idx0, idx1, frac


def upsample_bilinear(grid: NDArray[Any], out_h: int, out_w: int) -> NDArray[Any]:
    """Bilinear upsample a ``(D, gh, gw)`` grid to ``(D, out_h, out_w)``.

    Smoother than :func:`upsample_nearest` but blends across patch boundaries,
    which can be undesirable where hard class edges matter. Pure NumPy, so the
    core stays torch-free.
    """
    arr = as_float(grid)
    if arr.ndim != 3:
        raise ValueError(f"expected (D, gh, gw), got shape {arr.shape}")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output size must be positive")
    _, gh, gw = arr.shape
    if gh == 0 or gw == 0:
        raise ValueError("input grid has a zero spatial dimension")
    r0, r1, rf = _axis_weights(out_h, gh)
    c0, c1, cf = _axis_weights(out_w, gw)
    # _axis_weights builds the fractional weights in float64; cast them to the
    # grid's float dtype so bilinear upsampling preserves a float32 input (as
    # upsample_nearest does) instead of silently promoting the whole tile to
    # float64 and doubling peak memory.
    rf = rf.astype(arr.dtype, copy=False)
    cf = cf.astype(arr.dtype, copy=False)
    # Interpolate along rows, then along columns (separable).
    top = arr[:, r0, :]
    bot = arr[:, r1, :]
    rows = top * (1.0 - rf)[None, :, None] + bot * rf[None, :, None]
    left = rows[:, :, c0]
    right = rows[:, :, c1]
    out: NDArray[Any] = left * (1.0 - cf)[None, None, :] + right * cf[None, None, :]
    return out


def upsample(
    grid: NDArray[Any], out_h: int, out_w: int, *, method: str = "nearest"
) -> NDArray[Any]:
    """Upsample a patch grid by ``method`` (``'nearest'`` or ``'bilinear'``)."""
    if method == "nearest":
        return upsample_nearest(grid, out_h, out_w)
    if method == "bilinear":
        return upsample_bilinear(grid, out_h, out_w)
    raise ValueError(f"unknown upsampling method {method!r}; expected one of {UPSAMPLING_METHODS}")


def tokens_to_grid(tokens: NDArray[Any], grid_h: int, grid_w: int) -> NDArray[Any]:
    """Reshape ``(n_tokens, D)`` patch tokens to a ``(D, grid_h, grid_w)`` grid.

    If a leading class/registers token makes ``n_tokens`` one (or more) larger
    than ``grid_h * grid_w``, the extra leading tokens are dropped.
    """
    arr = as_float(tokens)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_tokens, D), got shape {arr.shape}")
    n_tokens, dim = arr.shape
    expected = grid_h * grid_w
    if n_tokens < expected:
        raise ValueError(f"have {n_tokens} tokens, need {expected} for grid")
    patch = arr[n_tokens - expected :]  # drop leading non-patch tokens
    out: NDArray[Any] = patch.T.reshape(dim, grid_h, grid_w)
    return out


def fuse_feature_maps(
    maps: Sequence[NDArray[Any]],
    out_h: int,
    out_w: int,
    *,
    mode: str = "concat",
    method: str = "nearest",
) -> NDArray[Any]:
    """Resample each ``(C_i, h_i, w_i)`` feature map to the tile grid and combine.

    ``mode`` ``concat`` stacks channels (dimension = sum of channels); ``sum``
    requires equal channel counts and adds them. ``method`` is the spatial
    upsampling method (``'nearest'`` or ``'bilinear'``; see :func:`upsample`).
    """
    if not maps:
        raise ValueError("need at least one feature map")
    resized = [upsample(as_float(m), out_h, out_w, method=method) for m in maps]
    if mode == "concat":
        out: NDArray[Any] = np.concatenate(resized, axis=0)
        return out
    if mode == "sum":
        channels = {m.shape[0] for m in resized}
        if len(channels) != 1:
            raise ValueError("'sum' mode needs equal channel counts across maps")
        stacked: NDArray[Any] = np.sum(np.stack(resized, axis=0), axis=0)
        return stacked
    raise ValueError(f"unknown fuse mode {mode!r}; expected 'concat' or 'sum'")
