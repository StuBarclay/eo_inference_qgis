"""Stream windows through a batched embedder into a mosaic (pure orchestration).

This module owns the loop that drives the ``embed`` algorithm, but touches no
GDAL, torch or QGIS: the caller injects a ``read_window`` callable (E-1) and an
``embed_batch`` callable (E-2), so the windowing / batching / cancellation /
progress / mosaicking logic is fully unit-testable with NumPy alone.

Only one batch of tiles is materialised at a time — the reader hands back a
window-sized array, the batch is embedded, results are scattered into the
:class:`~eo_inference_toolbox.eocore.tiling.MosaicAccumulator`, and the buffer is
dropped before the next batch. Peak memory therefore tracks the batch/tile size
rather than the whole scene (the fix for H-5).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .descriptor import DEFAULT_NODATA, EmbeddingDescriptor
from .normalize import l2_normalize
from .tiling import MosaicAccumulator, Window

#: Read one window's pixels as a ``(C, h, w)`` array (h/w = window height/width).
ReadWindow = Callable[[Window], NDArray[Any]]
#: Embed a batch of ``(C, h, w)`` tiles to a same-length list of ``(D, h, w)`` cubes.
EmbedBatch = Callable[[list[NDArray[Any]]], list[NDArray[Any]]]
#: Per-window blend weight (scalar or ``(h, w)`` array) for overlap feathering (E-4).
WeightFn = Callable[[Window], float | NDArray[Any]]


def input_valid_mask(
    chip: NDArray[Any],
    *,
    nodata: float | None = None,
    detect_void: bool = False,
) -> NDArray[Any]:
    """Per-pixel validity mask for an input ``(C, h, w)`` imagery tile.

    Returns a ``(h, w)`` boolean array (``True`` = valid) marking which pixels
    should contribute to the embedding. A pixel is treated as **invalid** when:

    * any band is non-finite (NaN / Inf), or
    * ``nodata`` is given and *every* band equals it (the declared nodata fill —
      the "all bands" rule matches :func:`eocore.ml.valid_mask_from_nodata` and
      avoids nuking a real pixel that merely happens to hit the sentinel in one
      band), or
    * ``detect_void`` is set and *every* band is exactly ``0`` (the off-scene
      black corners of a rotated / reprojected scene, whose fill is 0 rather than
      a declared nodata). This is a heuristic — a genuinely all-zero valid pixel
      is vanishingly rare in reflectance imagery but possible — so it is opt-in.

    Off-scene / nodata pixels must not be embedded as if they were real ground:
    feeding a ``-9999`` (or black) fill through a ViT contaminates the valid
    patch neighbours via the receptive field, and the output pixel itself is then
    a meaningless vector that pollutes every downstream similarity / change /
    classify result. Callers pair this mask with :func:`sanitize_invalid` (to
    keep the fill out of the forward pass) and a zero mosaic weight (so the pixel
    lands as nodata in the output).
    """
    arr = np.asarray(chip, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (C, h, w) tile, got shape {arr.shape}")
    valid = np.all(np.isfinite(arr), axis=0)
    if nodata is not None and np.isfinite(nodata):
        valid = valid & ~np.all(arr == np.float32(nodata), axis=0)
    if detect_void:
        valid = valid & ~np.all(arr == np.float32(0.0), axis=0)
    return np.asarray(valid, dtype=bool)


def sanitize_invalid(chip: NDArray[Any], valid_mask: NDArray[Any]) -> NDArray[Any]:
    """Return a float32 copy of ``chip`` with invalid pixels imputed, not fed raw.

    Invalid pixels (``valid_mask`` ``False``) are replaced band-by-band with the
    mean of that band's *valid* pixels, so the model never sees a ``-9999`` / NaN
    / black-corner value that its receptive field would smear into the valid
    neighbours. If the whole tile is invalid the copy is zeroed (its embedding is
    discarded anyway, since every pixel gets zero mosaic weight). The input array
    is never mutated. When ``valid_mask`` is all ``True`` the pixel values are
    returned unchanged (only the dtype is normalised to float32).
    """
    arr = np.array(chip, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (C, h, w) tile, got shape {arr.shape}")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.all():
        return arr
    invalid = ~mask
    if mask.any():
        means = arr[:, mask].mean(axis=1).astype(np.float32)
        arr[:, invalid] = means[:, None]
    else:
        arr[:, invalid] = np.float32(0.0)
    return arr


def _mask_weight(weight: float | NDArray[Any], valid_mask: NDArray[Any]) -> NDArray[Any]:
    """Fold a validity mask into a per-window blend weight (0 at invalid pixels).

    Broadcasts a scalar or ``(h, w)`` weight against the ``(h, w)`` mask so an
    invalid pixel gets weight 0 and is therefore excluded from the mosaic
    coverage (and stamped nodata by :func:`finalize_embedding`).
    """
    return np.asarray(weight, dtype=np.float32) * valid_mask.astype(np.float32)


def run_windowed_embed(
    windows: Sequence[Window],
    height: int,
    width: int,
    read_window: ReadWindow,
    embed_batch: EmbedBatch,
    *,
    batch_size: int = 1,
    weight_fn: WeightFn | None = None,
    input_nodata: float | None = None,
    detect_void: bool = False,
    progress: Callable[[int], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> MosaicAccumulator | None:
    """Embed a raster window-by-window and mosaic the result.

    ``windows`` are the read windows covering the scene (from
    :func:`~eo_inference_toolbox.eocore.tiling.iter_windows`); ``height`` /
    ``width`` size the output mosaic. Windows are read and embedded in groups of
    ``batch_size`` (default 1 = one tile per forward pass, i.e. v0.1.0
    behaviour). ``weight_fn`` supplies an optional per-window taper for
    overlap feathering; ``progress`` receives an integer percentage; and
    ``is_canceled`` is polled once per batch.

    ``input_nodata`` and ``detect_void`` guard against off-scene / nodata input
    pixels being embedded as if they were real ground (see
    :func:`input_valid_mask`): when either is active, each tile's invalid pixels
    are imputed before the forward pass (:func:`sanitize_invalid`) and given zero
    mosaic weight, so they land as nodata in the output rather than as a
    meaningless vector. With both off (the default) the read/embed/accumulate
    path is byte-for-byte the pre-0.2.0 behaviour.

    Returns the filled :class:`MosaicAccumulator`, or ``None`` if cancelled.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    window_list = list(windows)
    total = len(window_list)
    if total == 0:
        raise ValueError("need at least one window")

    detect_invalid = input_nodata is not None or detect_void
    accumulator: MosaicAccumulator | None = None
    done = 0
    for start in range(0, total, batch_size):
        if is_canceled is not None and is_canceled():
            return None
        batch = window_list[start : start + batch_size]
        chips: list[NDArray[Any]] = []
        masks: list[NDArray[Any] | None] = []
        for window in batch:
            chip = np.ascontiguousarray(read_window(window))
            mask: NDArray[Any] | None = None
            if detect_invalid:
                candidate = input_valid_mask(chip, nodata=input_nodata, detect_void=detect_void)
                if not candidate.all():
                    # Only impute / reweight when something is actually invalid, so
                    # a fully-valid tile stays byte-identical to the default path.
                    chip = sanitize_invalid(chip, candidate)
                    mask = candidate
            chips.append(chip)
            masks.append(mask)
        embeddings = embed_batch(chips)
        if len(embeddings) != len(batch):
            raise ValueError(
                f"embed_batch returned {len(embeddings)} outputs for {len(batch)} tiles"
            )
        for window, embedded, mask in zip(batch, embeddings, masks, strict=True):
            arr = np.asarray(embedded, dtype=np.float32)
            if accumulator is None:
                accumulator = MosaicAccumulator(int(arr.shape[0]), height, width)
            weight = weight_fn(window) if weight_fn is not None else 1.0
            if mask is not None:
                weight = _mask_weight(weight, mask)
            accumulator.add(window, arr, weight)
            done += 1
            if progress is not None:
                progress(int(100 * done / total))
    if accumulator is None:  # pragma: no cover - guarded by the empty-windows check
        raise ValueError("no tiles were produced")
    return accumulator


def finalize_embedding(
    mosaic: MosaicAccumulator,
    descriptor: EmbeddingDescriptor,
    *,
    nodata: float = DEFAULT_NODATA,
) -> tuple[NDArray[Any], NDArray[Any], EmbeddingDescriptor]:
    """Turn a filled mosaic into the embedding cube, its coverage mask and descriptor.

    The shared tail of the ``embed`` pipeline, factored out of
    :mod:`eo_inference_toolbox.algorithms.embed` (M7) so the exact sequence is
    unit-tested here rather than reimplemented by the test. It: (1) reads the
    overlap-averaged mosaic with uncovered pixels filled with 0 so normalisation
    is well defined; (2) L2-normalises every pixel vector — all three backends
    emit non-unit vectors, and the similarity/change metrics' cosine==dot-product
    shortcut is only valid once the raster is unit length; (3) stamps ``nodata``
    into pixels that no tile covered; and (4) replaces the descriptor's band count
    with the true output depth and marks it unit-normalised.

    Returns ``(normalized, coverage, descriptor)``. This is pure NumPy — no GDAL,
    torch or QGIS — so both the algorithm and its orchestration test call it.
    """
    coverage = mosaic.coverage
    normalized = l2_normalize(mosaic.result(fill=0.0))
    # A backend that emits NaN/Inf for a tile would otherwise land non-finite
    # values in the embedding raster: such a pixel is "covered" (weight > 0), so
    # the uncovered-stamp below misses it. Fold finiteness into coverage so any
    # non-finite output pixel is written as nodata and excluded from coverage.
    finite = np.all(np.isfinite(normalized), axis=0)
    coverage = coverage & finite
    normalized[:, ~coverage] = nodata
    final = dataclasses.replace(
        descriptor, n_bands=int(normalized.shape[0]), is_unit_normalized=True
    )
    return normalized, coverage, final
