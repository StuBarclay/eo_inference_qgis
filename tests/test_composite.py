"""Tests for nodata-aware temporal compositing (pure NumPy, no GDAL)."""

from __future__ import annotations

import numpy as np
import pytest

from eo_inference_toolbox.eocore.composite import COMPOSITE_MODES, composite


def test_median_ignores_nodata_per_pixel() -> None:
    # Three 1-band 1x1 scenes: values 10, nodata(0), 20 => median of {10,20}=15.
    scenes = [
        np.array([[[10.0]]]),
        np.array([[[0.0]]]),  # nodata
        np.array([[[20.0]]]),
    ]
    out = composite(scenes, nodata=0.0, mode="median")
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == pytest.approx(15.0)


def test_mean_ignores_nodata() -> None:
    scenes = [np.array([[[4.0]]]), np.array([[[0.0]]]), np.array([[[8.0]]])]
    out = composite(scenes, nodata=0.0, mode="mean")
    assert out[0, 0, 0] == pytest.approx(6.0)


def test_all_nodata_pixel_is_filled() -> None:
    scenes = [np.array([[[0.0]]]), np.array([[[0.0]]])]
    out = composite(scenes, nodata=0.0, mode="median")
    assert out[0, 0, 0] == pytest.approx(0.0)


def test_first_valid_takes_earliest_valid_scene() -> None:
    # Pixel A valid first in scene 0; pixel B nodata in scene 0, valid in scene 1.
    s0 = np.array([[[5.0, 0.0]]])  # band0: [A=5, B=nodata]
    s1 = np.array([[[7.0, 9.0]]])
    out = composite([s0, s1], nodata=0.0, mode="first_valid")
    assert out[0, 0, 0] == pytest.approx(5.0)  # earliest valid
    assert out[0, 0, 1] == pytest.approx(9.0)  # first scene was nodata


def test_first_valid_none_valid_filled() -> None:
    s0 = np.array([[[0.0]]])
    s1 = np.array([[[0.0]]])
    out = composite([s0, s1], nodata=0.0, mode="first_valid")
    assert out[0, 0, 0] == pytest.approx(0.0)


def test_nan_treated_as_invalid() -> None:
    scenes = [np.array([[[np.nan]]]), np.array([[[3.0]]])]
    out = composite(scenes, nodata=-1.0, mode="median")
    assert out[0, 0, 0] == pytest.approx(3.0)


def test_composite_validates_mode_and_shapes() -> None:
    with pytest.raises(ValueError, match="unknown composite mode"):
        composite([np.zeros((1, 2, 2))], mode="bogus")
    with pytest.raises(ValueError, match="same"):
        composite([np.zeros((1, 2, 2)), np.zeros((1, 3, 3))])
    with pytest.raises(ValueError, match="at least one"):
        composite([])
    assert set(COMPOSITE_MODES) == {"median", "mean", "first_valid"}
