"""Tests for tiling windows and the overlap-averaging mosaic accumulator."""

from __future__ import annotations

import numpy as np
import pytest

from eo_inference_toolbox.eocore.tiling import (
    MosaicAccumulator,
    Window,
    cosine_taper_weights,
    iter_windows,
)


def test_single_window_when_smaller_than_tile() -> None:
    windows = list(iter_windows(width=100, height=80, tile=256))
    assert len(windows) == 1
    assert windows[0] == Window(col_off=0, row_off=0, width=100, height=80)


def test_windows_cover_every_pixel() -> None:
    width, height, tile = 500, 300, 256
    covered = np.zeros((height, width), dtype=bool)
    for win in iter_windows(width, height, tile, overlap=32):
        covered[win.row_off : win.row_end, win.col_off : win.col_end] = True
    assert covered.all()


def test_interior_windows_are_full_size() -> None:
    # Edge windows shift flush rather than clipping, so all chips are tile-sized.
    for win in iter_windows(600, 600, 256, overlap=0):
        assert win.width == 256
        assert win.height == 256


def test_iter_windows_validates_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        list(iter_windows(100, 100, 32, overlap=32))


def test_iter_windows_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        list(iter_windows(0, 10, 8))


def test_mosaic_accumulator_averages_overlap() -> None:
    acc = MosaicAccumulator(n_bands=1, height=4, width=8)
    left = Window(col_off=0, row_off=0, width=5, height=4)
    right = Window(col_off=3, row_off=0, width=5, height=4)
    acc.add(left, np.ones((1, 4, 5), dtype=np.float32))
    acc.add(right, np.full((1, 4, 5), 3.0, dtype=np.float32))
    result = acc.result()
    # Non-overlap left = 1, overlap columns 3-4 = mean(1,3)=2, non-overlap right = 3.
    assert result[0, 0, 0] == pytest.approx(1.0)
    assert result[0, 0, 4] == pytest.approx(2.0)
    assert result[0, 0, 7] == pytest.approx(3.0)


def test_mosaic_coverage_flags_unseen() -> None:
    acc = MosaicAccumulator(n_bands=2, height=4, width=4)
    acc.add(Window(col_off=0, row_off=0, width=2, height=2), np.ones((2, 2, 2)))
    coverage = acc.coverage
    assert coverage[0, 0]
    assert not coverage[3, 3]


def test_mosaic_add_shape_check() -> None:
    acc = MosaicAccumulator(n_bands=3, height=4, width=4)
    with pytest.raises(ValueError, match="tile shape"):
        acc.add(Window(col_off=0, row_off=0, width=2, height=2), np.ones((3, 2, 3)))


def test_cosine_taper_is_strictly_positive_and_peaks_in_centre() -> None:
    w = cosine_taper_weights(16, 16, taper=4)
    assert w.shape == (16, 16)
    assert np.all(w > 0.0)  # a solo-covered edge pixel must still contribute
    assert w.max() == pytest.approx(1.0)
    # Centre is the plateau (==1); corners are the lightest.
    assert w[8, 8] == pytest.approx(1.0)
    assert w[0, 0] < w[8, 8]
    assert w[0, 0] == pytest.approx(w[-1, -1])  # symmetric


def test_cosine_taper_zero_returns_uniform() -> None:
    w = cosine_taper_weights(5, 7, taper=0)
    assert np.allclose(w, 1.0)


def test_mosaic_accepts_array_weight_and_reconstructs_solo_pixels() -> None:
    # A single window covering the whole grid with a feather weight must
    # reconstruct the tile exactly (weighted mean of one contributor = itself).
    acc = MosaicAccumulator(n_bands=1, height=6, width=6)
    tile = np.arange(36, dtype=np.float32).reshape(1, 6, 6)
    weight = cosine_taper_weights(6, 6, taper=2)
    acc.add(Window(col_off=0, row_off=0, width=6, height=6), tile, weight)
    assert acc.coverage.all()
    assert np.allclose(acc.result(), tile, atol=1e-4)


def test_mosaic_array_weight_shape_check() -> None:
    acc = MosaicAccumulator(n_bands=1, height=4, width=4)
    with pytest.raises(ValueError, match="weight shape"):
        acc.add(
            Window(col_off=0, row_off=0, width=2, height=2),
            np.ones((1, 2, 2)),
            np.ones((3, 3)),
        )


def test_feathered_overlap_biases_toward_nearer_tile() -> None:
    # Two half-overlapping tiles with constant but different values. With a
    # feather taper the overlap is a smooth blend, not a flat 50/50 step.
    acc = MosaicAccumulator(n_bands=1, height=4, width=8)
    left_win = Window(col_off=0, row_off=0, width=6, height=4)
    right_win = Window(col_off=2, row_off=0, width=6, height=4)
    acc.add(left_win, np.ones((1, 4, 6), dtype=np.float32), cosine_taper_weights(4, 6, taper=3))
    acc.add(
        right_win, np.full((1, 4, 6), 3.0, dtype=np.float32), cosine_taper_weights(4, 6, taper=3)
    )
    row = acc.result()[0, 0]
    overlap = row[2:6]
    # Blend stays within the two source values and trends upward toward the right.
    assert np.all(overlap >= 1.0) and np.all(overlap <= 3.0)
    assert overlap[-1] > overlap[0]
