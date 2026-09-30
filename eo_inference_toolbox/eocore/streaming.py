"""Stream a large embedding raster through a per-pixel analytic (pure orchestration).

The analytics algorithms — similarity, change, classify, cluster and
embedding→RGB — otherwise read the whole embedding cube into RAM. Embedding
cubes are the widest rasters in the pipeline (``D`` = 512-1024 float32 bands),
so a large AOI can exhaust memory even though every analytic's *output* is only
one to three bands. These helpers let an algorithm process the raster one tile at
a time: only one tile of the wide input is materialised, while the small output
is assembled in full and written once.

Like :mod:`~eo_inference_toolbox.eocore.runner`, this module touches no GDAL,
torch or QGIS — the caller injects a ``read_window`` callable (e.g. from
:func:`eo_inference_toolbox.algorithms._raster.open_window_reader`), so the
windowing / assembly / sampling logic is fully unit-tested with NumPy alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .tiling import Window

#: Read one window's pixels as a ``(bands, h, w)`` array.
ReadWindow = Callable[[Window], NDArray[Any]]
#: Map a ``(bands, h, w)`` input tile to a ``(out_bands, h, w)`` output tile.
BlockFn = Callable[[NDArray[Any]], NDArray[Any]]
#: Boolean ``(h, w)`` valid-pixel mask for one input tile.
MaskFn = Callable[[NDArray[Any]], NDArray[Any]]

#: Default streaming tile edge (pixels). A ``D``-band float32 tile of this size
#: peaks at roughly ``D * TILE**2 * 4`` bytes regardless of the scene size — the
#: whole point of the low-memory path.
DEFAULT_STREAM_TILE = 256

#: Default whole-cube allocation the auto low-memory decision tolerates before it
#: switches an analytic to the streaming path (megabytes). Chosen so the fast
#: whole-cube path stays the default for ordinary AOIs and only large embedding
#: cubes — the widest rasters in the pipeline — trip the streamer. Tunable per run
#: through each algorithm's advanced "Auto low-memory budget" parameter.
DEFAULT_MEMORY_BUDGET_MB = 1024


def estimate_cube_bytes(
    width: int, height: int, n_bands: int, *, itemsize: int = 4, n_cubes: int = 1
) -> int:
    """Bytes needed to hold ``n_cubes`` dense ``(n_bands, height, width)`` cubes.

    A pure size estimate for the *input* the whole-cube analytic path would
    materialise at once (``itemsize`` defaults to 4 for float32; ``n_cubes`` > 1
    covers change, which holds every raster in the series simultaneously). It
    deliberately ignores the smaller transient copies (the ``(H*W, bands)`` matrix,
    the one-to-three-band output) so the figure is a simple, explainable lower
    bound the budget is compared against.
    """
    if width <= 0 or height <= 0 or n_bands <= 0:
        raise ValueError("width, height and n_bands must be positive")
    if itemsize <= 0 or n_cubes <= 0:
        raise ValueError("itemsize and n_cubes must be positive")
    return int(width) * int(height) * int(n_bands) * int(itemsize) * int(n_cubes)


def should_auto_stream(
    width: int,
    height: int,
    n_bands: int,
    *,
    budget_bytes: int,
    itemsize: int = 4,
    n_cubes: int = 1,
) -> bool:
    """True when the whole-cube allocation would exceed ``budget_bytes``.

    The decision behind the opt-in "auto low-memory" mode: compare the estimated
    whole-cube footprint (:func:`estimate_cube_bytes`) against a byte budget. A
    non-positive ``budget_bytes`` disables the auto path (always ``False``), so a
    user can turn the feature off by zeroing the budget. This never *forces*
    streaming — it only reports whether the input is large enough to warrant it;
    the caller still ORs it with the explicit checkbox.
    """
    if budget_bytes <= 0:
        return False
    return estimate_cube_bytes(width, height, n_bands, itemsize=itemsize, n_cubes=n_cubes) > int(
        budget_bytes
    )


def plan_windows(width: int, height: int, tile: int = DEFAULT_STREAM_TILE) -> list[Window]:
    """Non-overlapping tiles covering the raster exactly once (last row/col partial).

    Unlike :func:`~eo_inference_toolbox.eocore.tiling.iter_windows` (which shifts
    the final chip flush against the edge so a fixed-input model always sees a
    full tile), these tiles never overlap, so a per-pixel analytic writes every
    output pixel exactly once.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile <= 0:
        raise ValueError("tile must be positive")
    windows: list[Window] = []
    for row in range(0, height, tile):
        win_h = min(tile, height - row)
        for col in range(0, width, tile):
            win_w = min(tile, width - col)
            windows.append(Window(col_off=col, row_off=row, width=win_w, height=win_h))
    return windows


def map_windows(
    windows: Sequence[Window],
    read_window: ReadWindow,
    block_fn: BlockFn,
    *,
    out_bands: int,
    height: int,
    width: int,
    dtype: Any = np.float32,
    fill: float = 0.0,
    progress: Callable[[int], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> NDArray[Any] | None:
    """Assemble a small ``(out_bands, height, width)`` output by streaming input tiles.

    Each window is read into a ``(bands, h, w)`` array, handed to ``block_fn``
    (which returns the matching ``(out_bands, h, w)`` output block), and scattered
    into a pre-allocated output. Only one input tile is held at a time, so peak
    memory tracks the tile size, not the scene.

    ``progress`` receives an integer percentage; ``is_canceled`` is polled once
    per window. Returns the filled array, or ``None`` if cancelled.
    """
    if out_bands <= 0 or height <= 0 or width <= 0:
        raise ValueError("out_bands, height and width must be positive")
    window_list = list(windows)
    total = len(window_list)
    if total == 0:
        raise ValueError("need at least one window")
    out = np.full((out_bands, height, width), fill, dtype=dtype)
    for done, window in enumerate(window_list, start=1):
        if is_canceled is not None and is_canceled():
            return None
        cube = np.asarray(read_window(window))
        block = np.asarray(block_fn(cube))
        expected = (out_bands, window.height, window.width)
        if block.shape != expected:
            raise ValueError(f"block shape {block.shape} != expected {expected}")
        out[:, window.row_off : window.row_end, window.col_off : window.col_end] = block.astype(
            dtype, copy=False
        )
        if progress is not None:
            progress(int(100 * done / total))
    return out


def map_windows_multi(
    windows: Sequence[Window],
    read_windows: Sequence[ReadWindow],
    block_fn: Callable[[list[NDArray[Any]]], NDArray[Any]],
    *,
    out_bands: int,
    height: int,
    width: int,
    dtype: Any = np.float32,
    fill: float = 0.0,
    progress: Callable[[int], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> NDArray[Any] | None:
    """Assemble a small output by streaming the *same* window from several readers.

    The multi-input twin of :func:`map_windows`, for analytics that combine two or
    more pixel-aligned rasters — e.g. change / trajectory / anomaly across an
    embedding time series, the pipeline's heaviest reader because it otherwise
    holds *N* wide cubes at once. Every reader must cover the same
    ``(height, width)`` grid. For each window the matching tile is read from each
    reader, the list of ``(bands, h, w)`` tiles is handed to ``block_fn`` (which
    returns the matching ``(out_bands, h, w)`` block), and the result is scattered
    into a pre-allocated output. Only one tile per reader is held at a time, so
    peak memory tracks the tile size times the reader count, not the whole scene.

    ``progress`` receives an integer percentage; ``is_canceled`` is polled once per
    window. Returns the filled array, or ``None`` if cancelled.
    """
    if out_bands <= 0 or height <= 0 or width <= 0:
        raise ValueError("out_bands, height and width must be positive")
    readers = list(read_windows)
    if not readers:
        raise ValueError("need at least one reader")
    window_list = list(windows)
    total = len(window_list)
    if total == 0:
        raise ValueError("need at least one window")
    out = np.full((out_bands, height, width), fill, dtype=dtype)
    for done, window in enumerate(window_list, start=1):
        if is_canceled is not None and is_canceled():
            return None
        tiles = [np.asarray(read(window)) for read in readers]
        block = np.asarray(block_fn(tiles))
        expected = (out_bands, window.height, window.width)
        if block.shape != expected:
            raise ValueError(f"block shape {block.shape} != expected {expected}")
        out[:, window.row_off : window.row_end, window.col_off : window.col_end] = block.astype(
            dtype, copy=False
        )
        if progress is not None:
            progress(int(100 * done / total))
    return out


def collect_valid_sample(
    windows: Sequence[Window],
    read_window: ReadWindow,
    valid_fn: MaskFn,
    *,
    max_samples: int,
    seed: int = 0,
) -> tuple[NDArray[Any], int]:
    """Sample up to ``max_samples`` valid pixel vectors while streaming the tiles.

    Uses random-key (bottom-k) reservoir sampling: every valid pixel is assigned a
    key from a seeded RNG in window order and the ``max_samples`` smallest keys are
    kept, so at most ``max_samples`` + one tile's worth of pixels are ever in
    memory and the draw is reproducible for a fixed raster + tiling. Returns
    ``(sample, n_valid)`` where ``sample`` is an ``(n_kept, bands)`` matrix
    (``n_kept <= max_samples``) and ``n_valid`` is the total valid-pixel count.

    The selection is not the subset a whole-cube ``rng.choice`` would draw (this is
    documented behaviour of the opt-in low-memory path); it feeds only an
    unsupervised *fit*, never per-pixel output maths.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rng = np.random.default_rng(seed)
    keys: NDArray[Any] | None = None
    sample: NDArray[Any] | None = None
    n_valid = 0
    for window in windows:
        cube = np.asarray(read_window(window))
        mask = np.asarray(valid_fn(cube)).reshape(-1)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        n_valid += count
        vecs = cube.reshape(cube.shape[0], -1).T[mask]
        new_keys = rng.random(vecs.shape[0])
        if sample is None:
            keys, sample = new_keys, np.asarray(vecs)
        else:
            assert keys is not None
            keys = np.concatenate([keys, new_keys])
            sample = np.concatenate([sample, vecs], axis=0)
        if sample.shape[0] > max_samples:
            keep = np.argpartition(keys, max_samples)[:max_samples]
            keys, sample = keys[keep], sample[keep]
    if sample is None:
        return np.empty((0, 0), dtype=np.float32), 0
    return sample, n_valid


def partial_fit_stream(
    windows: Sequence[Window],
    read_window: ReadWindow,
    valid_fn: MaskFn,
    fit_batch: Callable[[NDArray[Any]], None],
    *,
    min_batch: int,
    progress: Callable[[int], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Feed every valid pixel to an incremental estimator in bounded batches.

    Streams the tiles, buffers each window's valid pixel vectors, and calls
    ``fit_batch`` (e.g. ``MiniBatchKMeans.partial_fit``) once the buffer reaches
    ``min_batch`` rows, flushing any remainder at the end. This lets the low-memory
    cluster path fit on the *whole* dataset — not a subsample — while never holding
    more than about ``min_batch`` + one tile of pixels in memory.

    Returns ``(n_valid, n_batches)``. Returns early with ``n_batches`` reflecting
    the work done if cancelled. The caller must ensure the total valid-pixel count
    is at least the number of clusters (an incremental K-means needs at least
    ``k`` samples for its first batch); this helper does not check that.
    """
    if min_batch <= 0:
        raise ValueError("min_batch must be positive")
    window_list = list(windows)
    total = len(window_list)
    if total == 0:
        raise ValueError("need at least one window")
    buffer: list[NDArray[Any]] = []
    buffer_rows = 0
    n_valid = 0
    n_batches = 0
    for done, window in enumerate(window_list, start=1):
        if is_canceled is not None and is_canceled():
            return n_valid, n_batches
        cube = np.asarray(read_window(window))
        mask = np.asarray(valid_fn(cube)).reshape(-1)
        count = int(np.count_nonzero(mask))
        if count:
            n_valid += count
            buffer.append(cube.reshape(cube.shape[0], -1).T[mask])
            buffer_rows += count
        if buffer_rows >= min_batch:
            fit_batch(np.concatenate(buffer, axis=0))
            n_batches += 1
            buffer, buffer_rows = [], 0
        if progress is not None:
            progress(int(100 * done / total))
    if buffer_rows > 0:
        fit_batch(np.concatenate(buffer, axis=0))
        n_batches += 1
    return n_valid, n_batches


def read_points(
    read_window: ReadWindow,
    rowcols: Sequence[tuple[int, int]],
    *,
    nodata: float,
) -> tuple[list[NDArray[Any]], list[int]]:
    """Read the bounding-box window covering ``rowcols`` once and pull valid vectors.

    Given a window-reader and a set of ``(row, col)`` grid cells, read the single
    window spanning them and return ``(vectors, kept)`` where ``kept`` are the
    indices into ``rowcols`` whose pixel was valid (all-finite and not equal to
    ``nodata`` in every band). Only that one bounding box is materialised, so
    sampling the pixels under a single vector feature never loads the whole cube.
    This is the memory-bounded twin of indexing an in-memory cube and is the
    unit-tested core of the classify algorithm's low-memory training collection.

    .. note::
       The window is the axis-aligned bounding box of ``rowcols``, so a set of
       cells that is sparse or spread across the raster still reads (but does not
       *retain*) every intervening pixel. It is intended for the compact cluster
       of grid cells under one vector feature; callers with cells scattered
       scene-wide should batch by locality (e.g. per feature) rather than pass
       one far-flung set, or the transient bounding-box read can approach the
       whole scene.
    """
    if not rowcols:
        return [], []
    rows = [rc[0] for rc in rowcols]
    cols = [rc[1] for rc in rowcols]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    block = np.asarray(
        read_window(Window(col_off=c0, row_off=r0, width=c1 - c0 + 1, height=r1 - r0 + 1))
    )
    vectors: list[NDArray[Any]] = []
    kept: list[int] = []
    for i, (row, col) in enumerate(rowcols):
        vec = block[:, row - r0, col - c0]
        if bool(np.all(np.isfinite(vec))) and bool(np.any(vec != nodata)):
            vectors.append(np.asarray(vec, dtype=np.float32))
            kept.append(i)
    return vectors, kept
