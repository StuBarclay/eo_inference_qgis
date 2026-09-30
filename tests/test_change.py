"""Tests for change / trajectory / anomaly metrics."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cube, unit_cube

from eo_inference_toolbox.eocore.change import (
    METRICS,
    anomaly_from_baseline,
    distance_map,
    mean_embedding,
    trajectory_magnitude,
)


def test_cosine_distance_zero_for_identical() -> None:
    cube = unit_cube(6, 4, 4, seed=10)
    dist = distance_map(cube, cube, metric="cosine", assume_unit=True)
    assert np.allclose(dist, 0.0, atol=1e-6)


def test_cosine_distance_clamped_to_range() -> None:
    # assume_unit=True trusts the caller, so non-unit inputs push the raw
    # 1 - dot outside [0, 2]; the clamp must hold the documented bounds.
    a = np.full((3, 1, 1), 2.0, dtype=np.float32)
    near = distance_map(a, a, metric="cosine", assume_unit=True)
    assert near[0, 0] == pytest.approx(0.0)  # raw 1 - 12 = -11 -> clamped to 0
    opposite = np.full((3, 1, 1), -2.0, dtype=np.float32)
    far = distance_map(a, opposite, metric="cosine", assume_unit=True)
    assert far[0, 0] == pytest.approx(2.0)  # raw 1 - (-12) = 13 -> clamped to 2
    assert near.min() >= 0.0
    assert far.max() <= 2.0


def test_euclidean_distance_matches_manual() -> None:
    a = make_cube(4, 3, 3, seed=11)
    b = make_cube(4, 3, 3, seed=12)
    got = distance_map(a, b, metric="euclidean")
    expected = np.sqrt(np.sum((a - b) ** 2, axis=0))
    assert np.allclose(got, expected, atol=1e-5)


def test_distance_map_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        distance_map(make_cube(4, 3, 3), make_cube(4, 3, 4))


def test_distance_map_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        distance_map(make_cube(2, 2, 2), make_cube(2, 2, 2), metric="manhattan")
    assert set(METRICS) == {"cosine", "euclidean"}


def test_mean_embedding_averages() -> None:
    a = np.ones((3, 2, 2), dtype=np.float32)
    b = np.full((3, 2, 2), 3.0, dtype=np.float32)
    out = mean_embedding([a, b])
    assert np.allclose(out, 2.0)


def test_mean_embedding_requires_cubes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_embedding([])


def test_mean_embedding_nodata_none_averages_sentinels_in() -> None:
    # Default (nodata=None) is the unchanged plain element-wise mean: an all-band
    # -9999 fill in one cube drags the average down, byte-for-byte as before.
    a = np.full((2, 1, 1), -9999.0, dtype=np.float32)
    b = np.full((2, 1, 1), 1.0, dtype=np.float32)
    plain = mean_embedding([a, b])
    np.testing.assert_array_equal(plain, np.mean(np.stack([a, b], axis=0), axis=0))
    assert np.allclose(plain, (-9999.0 + 1.0) / 2.0)


def test_mean_embedding_nodata_excludes_fill_pixels() -> None:
    # With nodata set, an all-band fill pixel is dropped from the per-pixel mean.
    nod = -9999.0
    a = np.array([[[nod, 1.0]], [[nod, 1.0]]], dtype=np.float32)  # (2, 1, 2)
    b = np.array([[[3.0, 7.0]], [[3.0, 7.0]]], dtype=np.float32)
    out = mean_embedding([a, b], nodata=nod)
    assert out.shape == (2, 1, 2)
    assert np.allclose(out[:, 0, 0], 3.0)  # a is fill here -> mean over {b}
    assert np.allclose(out[:, 0, 1], 4.0)  # mean(1, 7)


def test_mean_embedding_nodata_all_missing_stays_nodata() -> None:
    nod = -9999.0
    a = np.full((3, 1, 1), nod, dtype=np.float32)
    b = np.full((3, 1, 1), nod, dtype=np.float32)
    out = mean_embedding([a, b], nodata=nod)
    assert np.allclose(out, nod)


def test_mean_embedding_nodata_one_band_hit_is_not_fill() -> None:
    # Only ONE band equals the sentinel -> the pixel is real (all-band rule) and
    # is averaged normally, sentinel included on that single band.
    nod = -9999.0
    a = np.array([[[nod]], [[1.0]]], dtype=np.float32)  # band0 == fill, band1 == 1
    b = np.array([[[2.0]], [[3.0]]], dtype=np.float32)
    out = mean_embedding([a, b], nodata=nod)
    assert out[0, 0, 0] == pytest.approx((nod + 2.0) / 2.0)
    assert out[1, 0, 0] == pytest.approx(2.0)


def test_trajectory_is_sum_of_consecutive_distances() -> None:
    a = make_cube(4, 3, 3, seed=20)
    b = make_cube(4, 3, 3, seed=21)
    c = make_cube(4, 3, 3, seed=22)
    got = trajectory_magnitude([a, b, c], metric="euclidean")
    step1 = distance_map(a, b, metric="euclidean")
    step2 = distance_map(b, c, metric="euclidean")
    assert np.allclose(got, step1 + step2, atol=1e-5)


def test_trajectory_needs_two_cubes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        trajectory_magnitude([make_cube(3, 2, 2)])


def test_anomaly_from_baseline_equals_distance() -> None:
    cube = make_cube(4, 3, 3, seed=30)
    base = make_cube(4, 3, 3, seed=31)
    got = anomaly_from_baseline(cube, base, metric="euclidean")
    assert np.allclose(got, distance_map(cube, base, metric="euclidean"))
