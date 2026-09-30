"""Sample embedding vectors under vector features (training + reference points).

Turns a point / polygon layer plus a loaded embedding cube into the arrays the
classify and similarity algorithms need. Feature geometries are reprojected into
the cube's grid CRS first, so the training layer need not match the raster CRS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
)

from ..eocore.grid import GridSpec
from ..eocore.streaming import read_points
from ..eocore.tiling import Window

# Guard against a runaway polygon flooding memory with samples.
_MAX_PIXELS_PER_FEATURE = 500_000


def _geometry_kind(geom: QgsGeometry) -> str:
    gtype = QgsWkbTypes.geometryType(geom.wkbType())
    if gtype == QgsWkbTypes.GeometryType.PolygonGeometry:
        return "polygon"
    if gtype == QgsWkbTypes.GeometryType.LineGeometry:
        return "line"
    return "point"


def _pixel_valid(vector: NDArray[Any], nodata: float) -> bool:
    if not np.all(np.isfinite(vector)):
        return False
    return bool(np.any(vector != nodata))


def _point_rowcols(geom: QgsGeometry, grid: GridSpec) -> list[tuple[int, int]]:
    """Grid (row, col) for each vertex of a point/multipoint geometry, in-grid."""
    out: list[tuple[int, int]] = []
    for vertex in geom.vertices():
        row, col = grid.rowcol_for_xy(vertex.x(), vertex.y())
        if grid.contains_rowcol(row, col):
            out.append((row, col))
    return out


def _polygon_rowcols(geom: QgsGeometry, grid: GridSpec) -> list[tuple[int, int]]:
    """Grid (row, col) of every pixel centre inside a polygon geometry."""
    box = geom.boundingBox()
    top_row, left_col = grid.rowcol_for_xy(box.xMinimum(), box.yMaximum())
    bottom_row, right_col = grid.rowcol_for_xy(box.xMaximum(), box.yMinimum())
    row_lo = max(0, min(top_row, bottom_row))
    row_hi = min(grid.height - 1, max(top_row, bottom_row))
    col_lo = max(0, min(left_col, right_col))
    col_hi = min(grid.width - 1, max(left_col, right_col))
    out: list[tuple[int, int]] = []
    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            x, y = grid.xy_center(row, col)
            if geom.contains(QgsPointXY(x, y)):
                out.append((row, col))
                if len(out) >= _MAX_PIXELS_PER_FEATURE:
                    return out
    return out


def _rowcols_for_geometry(geom: QgsGeometry, grid: GridSpec) -> list[tuple[int, int]]:
    kind = _geometry_kind(geom)
    if kind == "polygon":
        return _polygon_rowcols(geom, grid)
    if kind == "line":
        # Sample the pixels the line's vertices fall in (a coarse but cheap proxy).
        return _point_rowcols(geom, grid)
    return _point_rowcols(geom, grid)


def _make_transform(
    source_crs: QgsCoordinateReferenceSystem,
    dst_crs: QgsCoordinateReferenceSystem,
    transform_context: QgsCoordinateTransformContext,
) -> QgsCoordinateTransform | None:
    if not source_crs.isValid() or source_crs == dst_crs:
        return None
    return QgsCoordinateTransform(source_crs, dst_crs, transform_context)


def sample_reference_rowcols(
    source: Any,
    grid: GridSpec,
    dst_crs: QgsCoordinateReferenceSystem,
    transform_context: QgsCoordinateTransformContext,
) -> list[tuple[int, int]]:
    """Collect in-grid (row, col) pixels under a reference point layer."""
    transform = _make_transform(source.sourceCrs(), dst_crs, transform_context)
    rowcols: list[tuple[int, int]] = []
    for feature in source.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        if transform is not None:
            geom = QgsGeometry(geom)
            geom.transform(transform)
        rowcols.extend(_rowcols_for_geometry(geom, grid))
    return rowcols


def collect_training_samples(
    cube: NDArray[Any],
    grid: GridSpec,
    source: Any,
    class_field: str,
    dst_crs: QgsCoordinateReferenceSystem,
    transform_context: QgsCoordinateTransformContext,
    *,
    nodata: float,
) -> tuple[NDArray[Any], list[Any]]:
    """Sample ``(n_samples, n_bands)`` training vectors and their class labels."""
    arr = np.asarray(cube)
    n_bands = arr.shape[0]
    transform = _make_transform(source.sourceCrs(), dst_crs, transform_context)
    vectors: list[NDArray[Any]] = []
    labels: list[Any] = []
    for feature in source.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        if transform is not None:
            geom = QgsGeometry(geom)
            geom.transform(transform)
        label = feature[class_field]
        if label is None:
            continue
        for row, col in _rowcols_for_geometry(geom, grid):
            vector = arr[:, row, col]
            if _pixel_valid(vector, nodata):
                vectors.append(np.asarray(vector, dtype=np.float32))
                labels.append(label)
    if not vectors:
        return np.zeros((0, n_bands), dtype=np.float32), []
    return np.stack(vectors, axis=0), labels


def collect_training_samples_windowed(
    read_window: Callable[[Window], NDArray[Any]],
    grid: GridSpec,
    source: Any,
    class_field: str,
    dst_crs: QgsCoordinateReferenceSystem,
    transform_context: QgsCoordinateTransformContext,
    *,
    n_bands: int,
    nodata: float,
) -> tuple[NDArray[Any], list[Any]]:
    """Like :func:`collect_training_samples` but never loads the whole cube.

    For each labelled feature it reads a single window spanning that feature's
    in-grid pixels (via ``read_window``) and samples the wanted cells within it, so
    peak memory is bounded by the largest single feature's bounding box rather than
    the whole scene. ``n_bands`` is supplied by the caller since there is no cube to
    infer it from. Results match the whole-cube collector pixel-for-pixel.
    """
    transform = _make_transform(source.sourceCrs(), dst_crs, transform_context)
    vectors: list[NDArray[Any]] = []
    labels: list[Any] = []
    for feature in source.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        if transform is not None:
            geom = QgsGeometry(geom)
            geom.transform(transform)
        label = feature[class_field]
        if label is None:
            continue
        rowcols = _rowcols_for_geometry(geom, grid)
        feat_vectors, _kept = read_points(read_window, rowcols, nodata=nodata)
        vectors.extend(feat_vectors)
        labels.extend([label] * len(feat_vectors))
    if not vectors:
        return np.zeros((0, n_bands), dtype=np.float32), []
    return np.stack(vectors, axis=0), labels
