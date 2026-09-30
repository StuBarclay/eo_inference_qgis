"""Tests for the low-memory streaming helpers (pure NumPy, no GDAL/torch/QGIS)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eo_inference_toolbox.eocore.streaming import (
    collect_valid_sample,
    estimate_cube_bytes,
    map_windows,
    map_windows_multi,
    partial_fit_stream,
    plan_windows,
    read_points,
    should_auto_stream,
)
from eo_inference_toolbox.eocore.tiling import Window


def _reader(cube: NDArray[Any]) -> Callable[[Window], NDArray[Any]]:
    """A fake windowed reader over an in-memory ``(bands, H, W)`` cube."""

    def read(window: Window) -> NDArray[Any]:
        return cube[
            :,
            window.row_off : window.row_end,
            window.col_off : window.col_end,
        ]

    return read


# --- plan_windows -----------------------------------------------------------


def test_plan_windows_cover_every_pixel_exactly_once() -> None:
    width, height, tile = 500, 300, 256
    counts = np.zeros((height, width), dtype=int)
    for win in plan_windows(width, height, tile):
        counts[win.row_off : win.row_end, win.col_off : win.col_end] += 1
    # Non-overlapping, complete cover: every pixel written exactly once.
    assert (counts == 1).all()


def test_plan_windows_last_tile_is_partial() -> None:
    windows = plan_windows(width=300, height=100, tile=256)
    # 2 columns (256 + 44) x 1 row (100).
    assert [(w.col_off, w.width) for w in windows] == [(0, 256), (256, 44)]
    assert {w.height for w in windows} == {100}


def test_plan_windows_single_window_when_smaller_than_tile() -> None:
    assert plan_windows(80, 60, 256) == [Window(0, 0, 80, 60)]


def test_plan_windows_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        plan_windows(0, 10, 16)
    with pytest.raises(ValueError, match="tile"):
        plan_windows(10, 10, 0)


# --- map_windows ------------------------------------------------------------


def test_map_windows_reassembles_identically() -> None:
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((5, 130, 170)).astype(np.float32)
    windows = plan_windows(170, 130, tile=64)

    # A per-pixel reduction: sum across bands -> single output band.
    def block_fn(tile: NDArray[Any]) -> NDArray[Any]:
        out: NDArray[Any] = tile.sum(axis=0, keepdims=True)
        return out

    out = map_windows(windows, _reader(cube), block_fn, out_bands=1, height=130, width=170)
    assert out is not None
    expected = cube.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_map_windows_is_tile_size_invariant() -> None:
    rng = np.random.default_rng(1)
    cube = rng.standard_normal((3, 97, 103)).astype(np.float32)

    def block_fn(tile: NDArray[Any]) -> NDArray[Any]:
        return (tile[:1] * 2.0).astype(np.float32)

    ref = map_windows(
        plan_windows(103, 97, 999),
        _reader(cube),
        block_fn,
        out_bands=1,
        height=97,
        width=103,
    )
    for tile in (16, 32, 64):
        out = map_windows(
            plan_windows(103, 97, tile),
            _reader(cube),
            block_fn,
            out_bands=1,
            height=97,
            width=103,
        )
        np.testing.assert_array_equal(out, ref)


def test_map_windows_progress_and_cancel() -> None:
    cube = np.ones((2, 40, 40), dtype=np.float32)
    windows = plan_windows(40, 40, tile=16)
    seen: list[int] = []
    out = map_windows(
        windows,
        _reader(cube),
        lambda t: t[:1],
        out_bands=1,
        height=40,
        width=40,
        progress=seen.append,
    )
    assert out is not None
    assert seen and seen[-1] == 100 and seen == sorted(seen)

    # Cancelling before any window returns None.
    cancelled = map_windows(
        windows,
        _reader(cube),
        lambda t: t[:1],
        out_bands=1,
        height=40,
        width=40,
        is_canceled=lambda: True,
    )
    assert cancelled is None


def test_map_windows_rejects_wrong_block_shape() -> None:
    cube = np.ones((2, 20, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="block shape"):
        map_windows(
            plan_windows(20, 20, 8),
            _reader(cube),
            lambda t: t,  # returns 2 bands, out_bands=1 expected
            out_bands=1,
            height=20,
            width=20,
        )


def test_map_windows_needs_a_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        map_windows([], _reader(np.ones((1, 4, 4))), lambda t: t, out_bands=1, height=4, width=4)


# --- map_windows_multi ------------------------------------------------------


def test_map_windows_multi_matches_whole_scene_computation() -> None:
    rng = np.random.default_rng(5)
    cube_a = rng.standard_normal((6, 90, 110)).astype(np.float32)
    cube_b = rng.standard_normal((6, 90, 110)).astype(np.float32)
    windows = plan_windows(110, 90, tile=32)

    # A per-pixel two-input reduction: summed absolute difference across bands.
    def block_fn(tiles: list[NDArray[Any]]) -> NDArray[Any]:
        a, b = tiles
        out: NDArray[Any] = np.abs(a - b).sum(axis=0, keepdims=True)
        return out

    streamed = map_windows_multi(
        windows, [_reader(cube_a), _reader(cube_b)], block_fn, out_bands=1, height=90, width=110
    )
    assert streamed is not None
    expected = np.abs(cube_a - cube_b).sum(axis=0, keepdims=True)
    # Per-pixel reductions are byte-identical no matter how the scene is tiled.
    np.testing.assert_array_equal(streamed, expected)


def test_map_windows_multi_is_tile_size_invariant() -> None:
    rng = np.random.default_rng(6)
    cubes = [rng.standard_normal((3, 71, 67)).astype(np.float32) for _ in range(3)]

    # Mean across the stack of aligned cubes, reduced over the band axis.
    def block_fn(tiles: list[NDArray[Any]]) -> NDArray[Any]:
        stacked = np.mean(np.stack(tiles, axis=0), axis=0)  # (bands, h, w)
        out: NDArray[Any] = stacked.mean(axis=0, keepdims=True)  # (1, h, w)
        return out

    readers = [_reader(c) for c in cubes]
    ref = map_windows_multi(
        plan_windows(67, 71, 999), readers, block_fn, out_bands=1, height=71, width=67
    )
    for tile in (8, 16, 64):
        out = map_windows_multi(
            plan_windows(67, 71, tile), readers, block_fn, out_bands=1, height=71, width=67
        )
        np.testing.assert_array_equal(out, ref)


def test_map_windows_multi_cancel_returns_none() -> None:
    cube = np.ones((2, 40, 40), dtype=np.float32)
    out = map_windows_multi(
        plan_windows(40, 40, 16),
        [_reader(cube), _reader(cube)],
        lambda tiles: tiles[0][:1],
        out_bands=1,
        height=40,
        width=40,
        is_canceled=lambda: True,  # cancelled before the first window
    )
    assert out is None


def test_map_windows_multi_validates_readers() -> None:
    with pytest.raises(ValueError, match="at least one reader"):
        map_windows_multi(
            plan_windows(4, 4, 2), [], lambda tiles: tiles[0][:1], out_bands=1, height=4, width=4
        )


# --- collect_valid_sample ---------------------------------------------------


def test_collect_valid_sample_counts_and_bounds() -> None:
    rng = np.random.default_rng(2)
    cube = rng.standard_normal((4, 50, 50)).astype(np.float32)
    # Mark a known number of pixels invalid (all-zero vector).
    cube[:, :10, :] = 0.0  # 500 invalid pixels of 2500

    def valid_fn(tile: NDArray[Any]) -> NDArray[Any]:
        out: NDArray[Any] = np.any(tile != 0.0, axis=0)
        return out

    sample, n_valid = collect_valid_sample(
        plan_windows(50, 50, 16), _reader(cube), valid_fn, max_samples=200, seed=0
    )
    assert n_valid == 2000
    assert sample.shape == (200, 4)  # capped at max_samples, band count preserved
    # Every kept vector is a real valid pixel (never an all-zero one).
    assert np.all(np.any(sample != 0.0, axis=1))


def test_collect_valid_sample_returns_all_when_under_cap() -> None:
    cube = np.ones((3, 8, 8), dtype=np.float32)
    sample, n_valid = collect_valid_sample(
        plan_windows(8, 8, 4),
        _reader(cube),
        lambda t: np.any(t != 0.0, axis=0),
        max_samples=10_000,
    )
    assert n_valid == 64
    assert sample.shape == (64, 3)


def test_collect_valid_sample_reproducible_for_fixed_tiling() -> None:
    rng = np.random.default_rng(3)
    cube = rng.standard_normal((2, 30, 30)).astype(np.float32)
    windows = plan_windows(30, 30, 16)
    a, _ = collect_valid_sample(
        windows, _reader(cube), lambda t: np.any(t != 0.0, axis=0), max_samples=50, seed=7
    )
    b, _ = collect_valid_sample(
        windows, _reader(cube), lambda t: np.any(t != 0.0, axis=0), max_samples=50, seed=7
    )
    np.testing.assert_array_equal(a, b)


def test_collect_valid_sample_empty_when_none_valid() -> None:
    cube = np.zeros((3, 12, 12), dtype=np.float32)
    sample, n_valid = collect_valid_sample(
        plan_windows(12, 12, 4),
        _reader(cube),
        lambda t: np.any(t != 0.0, axis=0),
        max_samples=100,
    )
    assert n_valid == 0
    assert sample.shape == (0, 0)


def test_collect_valid_sample_validates_max() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        collect_valid_sample(
            plan_windows(4, 4, 2),
            _reader(np.ones((1, 4, 4))),
            lambda t: np.any(t != 0.0, axis=0),
            max_samples=0,
        )


# --- partial_fit_stream -----------------------------------------------------


def test_partial_fit_stream_feeds_every_valid_pixel_once() -> None:
    rng = np.random.default_rng(4)
    cube = rng.standard_normal((3, 40, 40)).astype(np.float32)
    cube[:, :5, :] = 0.0  # 200 invalid pixels of 1600

    batches: list[NDArray[Any]] = []
    n_valid, n_batches = partial_fit_stream(
        plan_windows(40, 40, 16),
        _reader(cube),
        lambda t: np.any(t != 0.0, axis=0),
        batches.append,
        min_batch=256,
    )
    assert n_valid == 1400
    assert n_batches == len(batches) >= 1
    # Every valid vector reached the estimator exactly once, order preserved.
    fed = np.concatenate(batches, axis=0)
    assert fed.shape == (1400, 3)
    assert np.all(np.any(fed != 0.0, axis=1))
    # Only the final flush may be smaller than min_batch.
    assert all(b.shape[0] >= 256 for b in batches[:-1])


def test_partial_fit_stream_single_batch_when_under_min() -> None:
    cube = np.ones((2, 10, 10), dtype=np.float32)
    batches: list[NDArray[Any]] = []
    n_valid, n_batches = partial_fit_stream(
        plan_windows(10, 10, 4),
        _reader(cube),
        lambda t: np.any(t != 0.0, axis=0),
        batches.append,
        min_batch=10_000,
    )
    assert (n_valid, n_batches) == (100, 1)
    assert batches[0].shape == (100, 2)


def test_partial_fit_stream_cancel_stops_early() -> None:
    cube = np.ones((2, 40, 40), dtype=np.float32)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # allow the first window, then cancel

    batches: list[NDArray[Any]] = []
    n_valid, _ = partial_fit_stream(
        plan_windows(40, 40, 16),
        _reader(cube),
        lambda t: np.any(t != 0.0, axis=0),
        batches.append,
        min_batch=64,
        is_canceled=cancel,
    )
    assert n_valid < 1600  # did not process the whole scene


def test_partial_fit_stream_validates() -> None:
    with pytest.raises(ValueError, match="min_batch"):
        partial_fit_stream(
            plan_windows(4, 4, 2),
            _reader(np.ones((1, 4, 4))),
            lambda t: np.any(t != 0.0, axis=0),
            lambda b: None,
            min_batch=0,
        )


# --- read_points ------------------------------------------------------------


def test_read_points_matches_direct_cube_indexing() -> None:
    rng = np.random.default_rng(10)
    cube = rng.standard_normal((4, 50, 60)).astype(np.float32)
    rowcols = [(3, 7), (40, 55), (12, 12), (3, 55)]
    vectors, kept = read_points(_reader(cube), rowcols, nodata=-9999.0)
    assert kept == [0, 1, 2, 3]
    for (row, col), vec in zip(rowcols, vectors, strict=True):
        # The bbox-offset arithmetic must recover the exact pixel vector.
        np.testing.assert_array_equal(vec, cube[:, row, col])


def test_read_points_reads_only_the_bounding_box() -> None:
    cube = np.ones((3, 100, 100), dtype=np.float32)
    seen: list[Window] = []

    def recording_read(window: Window) -> NDArray[Any]:
        seen.append(window)
        return cube[:, window.row_off : window.row_end, window.col_off : window.col_end]

    rowcols = [(10, 20), (14, 25), (12, 22)]
    read_points(recording_read, rowcols, nodata=-9999.0)
    # Exactly one window, spanning just the points' bbox — not the whole scene.
    assert len(seen) == 1
    win = seen[0]
    assert (win.col_off, win.row_off, win.width, win.height) == (20, 10, 6, 5)


def test_read_points_skips_nodata_and_nonfinite() -> None:
    cube = np.ones((2, 10, 10), dtype=np.float32)
    cube[:, 2, 2] = -9999.0  # all-nodata pixel
    cube[0, 4, 4] = np.nan  # non-finite pixel
    rowcols = [(1, 1), (2, 2), (4, 4), (5, 5)]
    vectors, kept = read_points(_reader(cube), rowcols, nodata=-9999.0)
    # Indices 1 (nodata) and 2 (nan) are dropped; valid ones keep their positions.
    assert kept == [0, 3]
    assert len(vectors) == 2


def test_read_points_empty() -> None:
    assert read_points(_reader(np.ones((2, 4, 4))), [], nodata=-9999.0) == ([], [])


# --- estimate_cube_bytes / should_auto_stream -------------------------------


def test_estimate_cube_bytes_matches_numpy_nbytes() -> None:
    # The estimate must equal a real float32 cube's byte count for n_cubes=1.
    cube = np.zeros((7, 40, 50), dtype=np.float32)
    assert estimate_cube_bytes(width=50, height=40, n_bands=7) == cube.nbytes


def test_estimate_cube_bytes_scales_with_cubes_and_itemsize() -> None:
    one = estimate_cube_bytes(width=100, height=100, n_bands=8)
    assert estimate_cube_bytes(width=100, height=100, n_bands=8, n_cubes=3) == 3 * one
    # float64 (itemsize=8) doubles the float32 (itemsize=4) estimate.
    assert estimate_cube_bytes(width=100, height=100, n_bands=8, itemsize=8) == 2 * one


def test_estimate_cube_bytes_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_cube_bytes(width=0, height=10, n_bands=4)
    with pytest.raises(ValueError, match="positive"):
        estimate_cube_bytes(width=10, height=10, n_bands=4, n_cubes=0)


def test_should_auto_stream_trips_only_above_budget() -> None:
    # A 4-band 100x100 float32 cube is 160000 bytes.
    exact = estimate_cube_bytes(width=100, height=100, n_bands=4)
    assert exact == 160_000
    # Budget just below the cube -> stream; at/above -> keep the whole-cube path.
    assert should_auto_stream(100, 100, 4, budget_bytes=exact - 1) is True
    assert should_auto_stream(100, 100, 4, budget_bytes=exact) is False
    assert should_auto_stream(100, 100, 4, budget_bytes=exact + 1) is False


def test_should_auto_stream_counts_all_cubes() -> None:
    exact = estimate_cube_bytes(width=64, height=64, n_bands=6)
    # One cube fits the budget, but three of them (change's N-raster read) do not.
    assert should_auto_stream(64, 64, 6, budget_bytes=exact, n_cubes=1) is False
    assert should_auto_stream(64, 64, 6, budget_bytes=exact, n_cubes=3) is True


def test_should_auto_stream_disabled_by_nonpositive_budget() -> None:
    # A zero/negative budget switches the auto path off entirely (never streams).
    assert should_auto_stream(10_000, 10_000, 1024, budget_bytes=0) is False
    assert should_auto_stream(10_000, 10_000, 1024, budget_bytes=-1) is False
