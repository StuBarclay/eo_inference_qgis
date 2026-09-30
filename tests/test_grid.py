"""Tests for the pure grid / affine geometry helpers."""

from __future__ import annotations

import pytest

from eo_inference_toolbox.eocore.grid import (
    GridSpec,
    bboxes_intersect,
    grid_from_bounds,
    metres_to_degrees,
)


def test_geotransform_is_north_up() -> None:
    grid = GridSpec(min_x=100.0, max_y=200.0, pixel_size=10.0, width=5, height=4)
    gt = grid.geotransform
    assert gt == (100.0, 10.0, 0.0, 200.0, 0.0, -10.0)


def test_bounds_span_full_extent() -> None:
    grid = GridSpec(min_x=0.0, max_y=100.0, pixel_size=10.0, width=5, height=4)
    assert grid.bounds == (0.0, 60.0, 50.0, 100.0)


def test_xy_center_and_rowcol_roundtrip() -> None:
    grid = GridSpec(min_x=0.0, max_y=100.0, pixel_size=10.0, width=5, height=4)
    x, y = grid.xy_center(2, 3)
    assert grid.rowcol_for_xy(x, y) == (2, 3)


def test_contains_rowcol_bounds() -> None:
    grid = GridSpec(min_x=0.0, max_y=100.0, pixel_size=10.0, width=5, height=4)
    assert grid.contains_rowcol(3, 4)
    assert not grid.contains_rowcol(4, 0)
    assert not grid.contains_rowcol(0, 5)


def test_gridspec_validates() -> None:
    with pytest.raises(ValueError, match="pixel_size"):
        GridSpec(min_x=0.0, max_y=0.0, pixel_size=0.0, width=2, height=2)
    with pytest.raises(ValueError, match="width and height"):
        GridSpec(min_x=0.0, max_y=0.0, pixel_size=1.0, width=0, height=2)


def test_grid_from_bounds_covers_fully() -> None:
    grid = grid_from_bounds((0.0, 0.0, 95.0, 45.0), pixel_size=10.0)
    # ceil(95/10)=10 wide, ceil(45/10)=5 tall.
    assert (grid.width, grid.height) == (10, 5)
    assert grid.min_x == 0.0
    assert grid.max_y == 45.0


def test_grid_from_bounds_validates() -> None:
    with pytest.raises(ValueError, match="positive extent"):
        grid_from_bounds((10.0, 0.0, 0.0, 10.0), pixel_size=1.0)
    with pytest.raises(ValueError, match="pixel_size"):
        grid_from_bounds((0.0, 0.0, 10.0, 10.0), pixel_size=-1.0)


def test_bboxes_intersect() -> None:
    assert bboxes_intersect((0, 0, 10, 10), (5, 5, 15, 15))
    assert not bboxes_intersect((0, 0, 10, 10), (10, 10, 20, 20))  # touching only
    assert not bboxes_intersect((0, 0, 5, 5), (6, 6, 8, 8))


def test_metres_to_degrees_scales_by_111km() -> None:
    # A 10 m native resolution becomes a sub-milli-degree pixel, not "10 degrees":
    # this is the fix for defaulting the pixel size in a geographic CRS.
    assert metres_to_degrees(111_320.0) == pytest.approx(1.0)
    assert metres_to_degrees(10.0) == pytest.approx(10.0 / 111_320.0)
    assert 0.0 < metres_to_degrees(10.0) < 1e-3


def test_metres_to_degrees_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        metres_to_degrees(0.0)
    with pytest.raises(ValueError, match="positive"):
        metres_to_degrees(-5.0)
