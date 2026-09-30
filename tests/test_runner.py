"""Tests for the pure windowed / batched embed orchestration (E-1, E-2, E-4).

The runner is exercised with injected fakes — a ``read_window`` that fabricates
window-sized arrays and an ``embed_batch`` that echoes them — so the streaming,
batching, cancellation and feathering logic is verified without GDAL or torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eo_inference_toolbox.eocore.runner import (
    EmbedBatch,
    input_valid_mask,
    run_windowed_embed,
    sanitize_invalid,
)
from eo_inference_toolbox.eocore.tiling import Window, cosine_taper_weights, iter_windows


def _echo_batch(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
    """A fake backend batch: emit each ``(C, h, w)`` tile unchanged (D = C)."""
    return list(chips)


def _capturing_batch(store: list[NDArray[Any]]) -> EmbedBatch:
    """An echo batch that also records (a copy of) every chip the model receives."""

    def embed(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
        store.extend(np.array(chip) for chip in chips)
        return list(chips)

    return embed


def test_run_windowed_embed_reads_only_window_sized_arrays() -> None:
    width, height, tile = 20, 16, 8
    windows = list(iter_windows(width, height, tile, overlap=2))
    requested: list[tuple[int, int]] = []

    def read_window(window: Window) -> NDArray[Any]:
        requested.append((window.height, window.width))
        return np.ones((3, window.height, window.width), dtype=np.float32)

    mosaic = run_windowed_embed(windows, height, width, read_window, _echo_batch)

    assert mosaic is not None
    assert mosaic.result().shape == (3, height, width)
    # Every read is a single window; the full scene is never requested (E-1 / H-5).
    assert requested
    assert all(h <= tile and w <= tile for (h, w) in requested)
    assert (height, width) not in requested


def test_run_windowed_embed_default_batch_size_is_one() -> None:
    windows = list(iter_windows(32, 8, 8, overlap=0))  # a single row of 4 windows
    assert len(windows) == 4
    batch_lengths: list[int] = []

    def read_window(window: Window) -> NDArray[Any]:
        return np.ones((2, window.height, window.width), dtype=np.float32)

    def embed_batch(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
        batch_lengths.append(len(chips))
        return list(chips)

    run_windowed_embed(windows, 8, 32, read_window, embed_batch)
    assert batch_lengths == [1, 1, 1, 1]


def test_run_windowed_embed_batches_up_to_batch_size() -> None:
    windows = list(iter_windows(32, 8, 8, overlap=0))  # 4 windows
    batch_lengths: list[int] = []

    def read_window(window: Window) -> NDArray[Any]:
        return np.ones((2, window.height, window.width), dtype=np.float32)

    def embed_batch(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
        batch_lengths.append(len(chips))
        return list(chips)

    run_windowed_embed(windows, 8, 32, read_window, embed_batch, batch_size=3)
    assert batch_lengths == [3, 1]  # 4 windows in groups of 3


def test_run_windowed_embed_honours_cancellation() -> None:
    windows = list(iter_windows(32, 8, 8, overlap=0))
    seen = {"checks": 0}

    def is_canceled() -> bool:
        seen["checks"] += 1
        return seen["checks"] > 1  # allow the first batch, cancel before the second

    def read_window(window: Window) -> NDArray[Any]:
        return np.ones((1, window.height, window.width), dtype=np.float32)

    result = run_windowed_embed(windows, 8, 32, read_window, _echo_batch, is_canceled=is_canceled)
    assert result is None


def test_run_windowed_embed_reports_progress_monotonically() -> None:
    windows = list(iter_windows(24, 8, 8, overlap=0))  # 3 windows
    values: list[int] = []

    def read_window(window: Window) -> NDArray[Any]:
        return np.ones((1, window.height, window.width), dtype=np.float32)

    run_windowed_embed(windows, 8, 24, read_window, _echo_batch, progress=values.append)
    assert values == sorted(values)
    assert values[-1] == 100


def test_run_windowed_embed_applies_taper_weights() -> None:
    # Two horizontally-overlapping tiles carrying different constant values.
    width, height = 12, 4
    windows = [Window(0, 0, 8, 4), Window(4, 0, 8, 4)]

    def read_window(window: Window) -> NDArray[Any]:
        value = 1.0 if window.col_off == 0 else 3.0
        return np.full((1, window.height, window.width), value, dtype=np.float32)

    def taper(window: Window) -> NDArray[Any]:
        return cosine_taper_weights(window.height, window.width, taper=3)

    uniform = run_windowed_embed(windows, height, width, read_window, _echo_batch)
    feathered = run_windowed_embed(
        windows, height, width, read_window, _echo_batch, weight_fn=taper
    )
    assert uniform is not None and feathered is not None
    uni_row = uniform.result()[0, 0]
    fea_row = feathered.result()[0, 0]

    # Outside the overlap both agree with the source tiles.
    assert uni_row[0] == pytest.approx(1.0)
    assert uni_row[-1] == pytest.approx(3.0)
    assert fea_row[0] == pytest.approx(1.0)
    assert fea_row[-1] == pytest.approx(3.0)
    # Uniform weighting averages the overlap to a hard 2.0 everywhere ...
    overlap = slice(4, 8)
    assert np.allclose(uni_row[overlap], 2.0)
    # ... while feathering blends it into a gradient that is not flat 2.0.
    assert not np.allclose(fea_row[overlap], 2.0)
    assert fea_row[4] < fea_row[7]


def test_run_windowed_embed_validates_inputs() -> None:
    windows = list(iter_windows(8, 8, 8, overlap=0))

    def read_window(window: Window) -> NDArray[Any]:
        return np.ones((1, window.height, window.width), dtype=np.float32)

    with pytest.raises(ValueError, match="batch_size"):
        run_windowed_embed(windows, 8, 8, read_window, _echo_batch, batch_size=0)
    with pytest.raises(ValueError, match="at least one window"):
        run_windowed_embed([], 8, 8, read_window, _echo_batch)


def test_run_windowed_embed_rejects_batch_length_mismatch() -> None:
    windows = list(iter_windows(16, 8, 8, overlap=0))  # 2 windows

    def read_window(window: Window) -> NDArray[Any]:
        return np.ones((1, window.height, window.width), dtype=np.float32)

    def short_batch(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
        return list(chips)[:-1]  # drop one output

    with pytest.raises(ValueError, match="outputs for"):
        run_windowed_embed(windows, 8, 16, read_window, short_batch, batch_size=2)


# --- input validity: declared nodata + opt-in void (HIGH-1) -----------------


def test_input_valid_mask_flags_nodata_nonfinite_and_optin_void() -> None:
    chip = np.ones((2, 3, 4), dtype=np.float32)
    chip[:, 0, 0] = -9999.0  # every band == nodata -> invalid
    chip[0, 1, 1] = np.nan  # one band non-finite -> invalid
    chip[:, 2, 2] = 0.0  # every band zero -> void only when opted in

    mask = input_valid_mask(chip, nodata=-9999.0)
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert mask[2, 2]  # zeros stay valid unless detect_void
    # Everything else is valid.
    assert mask.sum() == mask.size - 2

    void_mask = input_valid_mask(chip, nodata=-9999.0, detect_void=True)
    assert not void_mask[2, 2]

    # A pixel where only ONE band hits the sentinel is still valid (all-band rule).
    one_band = np.ones((3, 1, 1), dtype=np.float32)
    one_band[0, 0, 0] = -9999.0
    assert input_valid_mask(one_band, nodata=-9999.0)[0, 0]

    # With no nodata declared and void off, only non-finite pixels are invalid.
    plain = input_valid_mask(chip)
    assert not plain[1, 1]
    assert plain[0, 0]  # -9999 is just data when nothing is declared


def test_input_valid_mask_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        input_valid_mask(np.ones((3, 3), dtype=np.float32), nodata=0.0)


def test_sanitize_invalid_imputes_band_mean_without_mutating_input() -> None:
    chip = np.ones((2, 2, 2), dtype=np.float32)
    chip[0, 0, 0] = 10.0  # band-0 valid pixels become [10, 1, 1] -> mean 4.0
    mask = np.ones((2, 2), dtype=bool)
    mask[1, 1] = False  # the one invalid pixel
    original = chip.copy()

    out = sanitize_invalid(chip, mask)

    np.testing.assert_array_equal(chip, original)  # input never mutated
    assert out[0, 1, 1] == pytest.approx(4.0)  # band-0 valid mean
    assert out[1, 1, 1] == pytest.approx(1.0)  # band-1 valid mean
    assert out[0, 0, 0] == 10.0  # valid pixels untouched
    assert out.dtype == np.float32


def test_sanitize_invalid_all_invalid_is_zeroed() -> None:
    chip = np.full((2, 2, 2), 7.0, dtype=np.float32)
    out = sanitize_invalid(chip, np.zeros((2, 2), dtype=bool))
    assert np.all(out == 0.0)


def test_sanitize_invalid_all_valid_returns_values_unchanged() -> None:
    chip = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2) + 1.0
    out = sanitize_invalid(chip, np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(out, chip)


def test_declared_nodata_block_becomes_uncovered_and_is_not_embedded() -> None:
    # One window covers the whole scene; a 2x3 all-band nodata block sits top-left.
    windows = list(iter_windows(8, 6, 8, overlap=0))
    assert len(windows) == 1

    def read_window(window: Window) -> NDArray[Any]:
        chip = np.ones((2, window.height, window.width), dtype=np.float32)
        chip[:, 0:2, 0:3] = -9999.0
        return chip

    seen: list[NDArray[Any]] = []
    mosaic = run_windowed_embed(
        windows, 6, 8, read_window, _capturing_batch(seen), input_nodata=-9999.0
    )
    assert mosaic is not None
    coverage = mosaic.coverage
    # The nodata block is excluded from coverage; everything else is covered.
    assert not coverage[0:2, 0:3].any()
    assert coverage[2:, :].all()
    assert coverage[0:2, 3:].all()
    # The model never saw the -9999 fill: it was imputed to the valid mean (1.0).
    assert len(seen) == 1
    assert not np.any(seen[0] == -9999.0)
    assert np.all(np.isfinite(seen[0]))
    assert np.allclose(seen[0][:, 0:2, 0:3], 1.0)


def test_detect_void_is_opt_in() -> None:
    windows = list(iter_windows(8, 6, 8, overlap=0))

    def read_window(window: Window) -> NDArray[Any]:
        chip = np.ones((2, window.height, window.width), dtype=np.float32)
        chip[:, 0:2, 0:3] = 0.0  # black corner, no declared nodata
        return chip

    # Off (default): all-zero pixels are treated as valid data -> fully covered.
    off = run_windowed_embed(windows, 6, 8, read_window, _echo_batch)
    assert off is not None
    assert off.coverage.all()

    # On: the black block is excluded from coverage.
    on = run_windowed_embed(windows, 6, 8, read_window, _echo_batch, detect_void=True)
    assert on is not None
    assert not on.coverage[0:2, 0:3].any()
    assert on.coverage[2:, :].all()


def test_guards_on_clean_scene_are_byte_identical() -> None:
    # A multi-window scene with only finite, positive, non-nodata pixels: turning
    # the guards on must not change a single output byte (or the coverage).
    windows = list(iter_windows(20, 16, 8, overlap=2))

    def read_window(window: Window) -> NDArray[Any]:
        base = np.arange(window.height * window.width, dtype=np.float32)
        ramp = base.reshape(window.height, window.width) + 1.0
        return np.stack([ramp, ramp + 100.0, ramp + 200.0], axis=0)

    plain = run_windowed_embed(windows, 16, 20, read_window, _echo_batch)
    guarded = run_windowed_embed(
        windows, 16, 20, read_window, _echo_batch, input_nodata=-9999.0, detect_void=True
    )
    assert plain is not None and guarded is not None
    np.testing.assert_array_equal(plain.result(), guarded.result())
    np.testing.assert_array_equal(plain.coverage, guarded.coverage)
