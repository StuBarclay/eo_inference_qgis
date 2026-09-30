"""Tests for the token/feature-map pooling and upsampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from eo_inference_toolbox.eocore.backends._pool import (
    UPSAMPLING_METHODS,
    fuse_feature_maps,
    tokens_to_grid,
    upsample,
    upsample_bilinear,
    upsample_nearest,
)


def test_nearest_upsample_repeats_blocks() -> None:
    grid = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    out = upsample_nearest(grid, 4, 4)
    assert out.shape == (1, 4, 4)
    # 2x integer scale => each source cell fills a 2x2 block.
    assert out[0, 0, 0] == 0.0
    assert out[0, 0, 3] == 1.0
    assert out[0, 3, 0] == 2.0
    assert out[0, 3, 3] == 3.0


def test_bilinear_upsample_identity_when_same_size() -> None:
    grid = np.random.default_rng(0).standard_normal((3, 5, 7)).astype(np.float32)
    out = upsample_bilinear(grid, 5, 7)
    assert np.allclose(out, grid, atol=1e-6)


def test_bilinear_upsample_interpolates_midpoints() -> None:
    # A 1x2 ramp [0, 1]; upsample to width 4. align_corners=False sample centres
    # sit at 0.125,0.375,0.625,0.875 in source space => clamped/interpolated.
    grid = np.array([[[0.0, 1.0]]], dtype=np.float32)
    out = upsample_bilinear(grid, 1, 4)
    assert out.shape == (1, 1, 4)
    # Monotonic increasing and bounded by the source range.
    row = out[0, 0]
    assert np.all(np.diff(row) > 0)
    assert row.min() >= 0.0 and row.max() <= 1.0


def test_bilinear_handles_singleton_axis() -> None:
    grid = np.array([[[5.0]]], dtype=np.float32)  # (1, 1, 1)
    out = upsample_bilinear(grid, 3, 4)
    assert out.shape == (1, 3, 4)
    assert np.allclose(out, 5.0)


def test_bilinear_upsample_preserves_input_float_dtype() -> None:
    # The interpolation weights are built in float64; casting them to the grid's
    # dtype keeps a float32 grid float32 (parity with upsample_nearest) instead of
    # silently promoting the whole tile to float64 and doubling peak memory.
    f32 = np.ones((1, 2, 2), dtype=np.float32)
    assert upsample_bilinear(f32, 4, 4).dtype == np.float32
    assert upsample_nearest(f32, 4, 4).dtype == np.float32  # the parity anchor
    # A genuine float64 grid is left float64 (the weights follow the input dtype,
    # they do not force it either way).
    f64 = np.ones((1, 2, 2), dtype=np.float64)
    assert upsample_bilinear(f64, 4, 4).dtype == np.float64


def test_upsample_dispatch_matches_named_helpers() -> None:
    grid = np.random.default_rng(1).standard_normal((2, 3, 3)).astype(np.float32)
    assert np.allclose(upsample(grid, 6, 6, method="nearest"), upsample_nearest(grid, 6, 6))
    assert np.allclose(upsample(grid, 6, 6, method="bilinear"), upsample_bilinear(grid, 6, 6))
    assert set(UPSAMPLING_METHODS) == {"nearest", "bilinear"}


def test_upsample_rejects_unknown_method() -> None:
    grid = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown upsampling method"):
        upsample(grid, 4, 4, method="cubic")


def test_upsample_validates_shape_and_size() -> None:
    with pytest.raises(ValueError, match="D, gh, gw"):
        upsample_bilinear(np.ones((2, 2)), 4, 4)
    with pytest.raises(ValueError, match="positive"):
        upsample_bilinear(np.ones((1, 2, 2)), 0, 4)


def test_tokens_to_grid_drops_leading_class_token() -> None:
    # 4 patch tokens + 1 leading class token, dim=3.
    tokens = np.arange(15, dtype=np.float32).reshape(5, 3)
    grid = tokens_to_grid(tokens, 2, 2)
    assert grid.shape == (3, 2, 2)


def test_fuse_feature_maps_concat_and_sum() -> None:
    a = np.ones((2, 2, 2), dtype=np.float32)
    b = np.full((3, 1, 1), 2.0, dtype=np.float32)
    concat = fuse_feature_maps([a, b], 2, 2, mode="concat")
    assert concat.shape == (5, 2, 2)
    summed = fuse_feature_maps([a, a], 2, 2, mode="sum")
    assert np.allclose(summed, 2.0)


def test_fuse_feature_maps_threads_upsampling_method() -> None:
    # A single coarse map fused to a finer grid must match the chosen upsampler.
    fmap = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    near = fuse_feature_maps([fmap], 4, 4, mode="concat", method="nearest")
    bilin = fuse_feature_maps([fmap], 4, 4, mode="concat", method="bilinear")
    assert np.allclose(near, upsample_nearest(fmap, 4, 4))
    assert np.allclose(bilin, upsample_bilinear(fmap, 4, 4))
    # The two methods genuinely differ on a non-constant map.
    assert not np.allclose(near, bilin)


def test_fuse_feature_maps_rejects_unknown_method() -> None:
    fmap = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown upsampling method"):
        fuse_feature_maps([fmap], 4, 4, method="cubic")
