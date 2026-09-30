"""Tests for the embedding -> RGB preview renderer."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cube

from eo_inference_toolbox.eocore.descriptor import DEFAULT_NODATA
from eo_inference_toolbox.eocore.rgb import (
    METHODS,
    apply_rgb_transform,
    embedding_to_rgb,
    fit_pca,
    fit_rgb_transform,
    stretch_values,
)
from eo_inference_toolbox.eocore.streaming import map_windows, plan_windows


def test_stretch_values_spans_byte_range() -> None:
    out = stretch_values(np.linspace(0.0, 1.0, 100), low_pct=0.0, high_pct=100.0)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_stretch_constant_channel_is_zero() -> None:
    out = stretch_values(np.full(50, 4.2))
    assert np.all(out == 0)


def test_stretch_empty() -> None:
    assert stretch_values(np.zeros(0)).shape == (0,)


def test_fit_pca_returns_components() -> None:
    matrix = make_cube(8, 20, 20, seed=50).reshape(8, -1).T
    mean, components = fit_pca(matrix, 3)
    assert mean.shape == (8,)
    assert components.shape == (3, 8)


def test_fit_pca_requires_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        fit_pca(np.zeros((0, 4)))


def test_fit_pca_rejects_non_finite() -> None:
    # A NaN/Inf in the fit sample would poison the SVD (non-finite components ->
    # an all-black preview); it must fail loudly instead.
    matrix = make_cube(6, 10, 10, seed=53).reshape(6, -1).T.copy()
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_pca(matrix, 3)
    matrix[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        fit_pca(matrix, 3)


def test_embedding_to_rgb_pca_shape_and_type() -> None:
    cube = make_cube(10, 16, 16, seed=51)
    rgb = embedding_to_rgb(cube, method="pca")
    assert rgb.shape == (3, 16, 16)
    assert rgb.dtype == np.uint8
    assert set(METHODS) == {"pca", "bands"}


def test_embedding_to_rgb_all_nodata_is_black() -> None:
    cube = np.full((6, 5, 5), DEFAULT_NODATA, dtype=np.float32)
    rgb = embedding_to_rgb(cube, method="pca")
    assert np.all(rgb == 0)


def test_embedding_to_rgb_bands_uses_triplet() -> None:
    cube = make_cube(7, 8, 8, seed=52)
    rgb = embedding_to_rgb(cube, method="bands", bands=[0, 2, 4])
    assert rgb.shape == (3, 8, 8)


def test_embedding_to_rgb_bad_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        embedding_to_rgb(make_cube(3, 4, 4), method="tsne")


def test_embedding_to_rgb_bad_triplet_raises() -> None:
    with pytest.raises(ValueError, match="three band indices"):
        embedding_to_rgb(make_cube(5, 4, 4), method="bands", bands=[0, 1])


@pytest.mark.parametrize("method,bands", [("pca", None), ("bands", [0, 3, 5])])
def test_fit_apply_matches_embedding_to_rgb(method: str, bands: list[int] | None) -> None:
    # The low-memory split (fit on the valid matrix, then apply) must reproduce
    # the one-shot whole-cube renderer exactly when fit on the same pixels.
    cube = make_cube(7, 24, 20, seed=60)
    n_bands = cube.shape[0]
    mask = np.any(cube != DEFAULT_NODATA, axis=0).reshape(-1)
    valid = cube.reshape(n_bands, -1).T[mask]
    transform = fit_rgb_transform(valid, method=method, bands=bands, n_bands=n_bands)
    tiled = apply_rgb_transform(cube, transform)
    np.testing.assert_array_equal(tiled, embedding_to_rgb(cube, method=method, bands=bands))


def test_apply_rgb_transform_is_tileable() -> None:
    # A fitted transform applied tile-by-tile reassembles the same image as
    # applying it to the whole cube — this is what the streaming preview relies on.
    cube = make_cube(6, 40, 30, seed=61)
    n_bands = cube.shape[0]
    mask = np.any(cube != DEFAULT_NODATA, axis=0).reshape(-1)
    valid = cube.reshape(n_bands, -1).T[mask]
    transform = fit_rgb_transform(valid, method="pca", n_bands=n_bands)
    whole = apply_rgb_transform(cube, transform)

    def read(win):  # type: ignore[no-untyped-def]
        return cube[:, win.row_off : win.row_end, win.col_off : win.col_end]

    tiled = map_windows(
        plan_windows(30, 40, 16),
        read,
        lambda tile: apply_rgb_transform(tile, transform),
        out_bands=3,
        height=40,
        width=30,
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(tiled, whole)
