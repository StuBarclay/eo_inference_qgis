"""GDAL-gated tests for the raster I/O bridge.

These skip automatically when GDAL is not importable (e.g. the CI lint sandbox),
and exercise a real GeoTIFF write -> read round-trip when it is present (e.g. the
GDAL container job, or inside QGIS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("osgeo.gdal")

from osgeo import osr

from eo_inference_toolbox.algorithms import _raster


def _wgs84_wkt() -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return str(srs.ExportToWkt())


def test_write_read_roundtrip(tmp_path: Path) -> None:
    cube = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    geotransform = (100.0, 1.0, 0.0, 50.0, 0.0, -1.0)
    path = str(tmp_path / "cube.tif")
    _raster.write_geotiff(
        path,
        cube,
        geotransform,
        _wgs84_wkt(),
        nodata=-9999.0,
        metadata={_raster.MD_BACKEND: "clay", _raster.MD_UNIT: "1"},
    )
    back = _raster.read_raster(path)
    assert back.cube.shape == (2, 4, 5)
    assert np.allclose(back.cube, cube)
    assert back.nodata == -9999.0
    assert _raster.read_metadata_item(path, _raster.MD_BACKEND) == "clay"


def test_grid_center_lonlat_is_identity_for_wgs84() -> None:
    from eo_inference_toolbox.eocore.grid import GridSpec

    grid = GridSpec(min_x=150.0, max_y=-33.0, pixel_size=0.01, width=10, height=10)
    lon, lat = _raster.grid_center_lonlat(grid, _wgs84_wkt())
    assert lon == pytest.approx(150.05, abs=1e-3)
    assert lat == pytest.approx(-33.05, abs=1e-3)


def _write_cube(path: str, cube: NDArray[np.float32], nodata: float | None = None) -> None:
    geotransform = (100.0, 1.0, 0.0, 50.0, 0.0, -1.0)
    _raster.write_geotiff(path, cube, geotransform, _wgs84_wkt(), nodata=nodata)


def test_read_raster_info_reads_header_only(tmp_path: Path) -> None:
    cube = np.arange(3 * 6 * 7, dtype=np.float32).reshape(3, 6, 7)
    path = str(tmp_path / "hdr.tif")
    _write_cube(path, cube, nodata=-1.0)
    info = _raster.read_raster_info(path)
    assert (info.n_bands, info.height, info.width) == (3, 6, 7)
    assert info.nodata == -1.0
    grid = info.grid_spec()
    assert (grid.width, grid.height) == (7, 6)
    assert grid.pixel_size == pytest.approx(1.0)


def test_open_window_reader_returns_only_window_blocks(tmp_path: Path) -> None:
    from eo_inference_toolbox.eocore.tiling import Window

    cube = np.arange(2 * 8 * 10, dtype=np.float32).reshape(2, 8, 10)
    path = str(tmp_path / "win.tif")
    _write_cube(path, cube)
    with _raster.open_window_reader(path) as read:
        block = read(Window(3, 2, 4, 5))  # col_off, row_off, width, height
    assert block.shape == (2, 5, 4)
    # The block must match the corresponding slice of the source cube.
    assert np.allclose(block, cube[:, 2:7, 3:7])


def test_open_window_reader_band_subset(tmp_path: Path) -> None:
    from eo_inference_toolbox.eocore.tiling import Window

    cube = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
    path = str(tmp_path / "bands.tif")
    _write_cube(path, cube)
    with _raster.open_window_reader(path, bands=[1, 3]) as read:  # 1-based: bands 0 and 2
        block = read(Window(0, 0, 5, 5))
    assert block.shape == (2, 5, 5)
    assert np.allclose(block[0], cube[0])
    assert np.allclose(block[1], cube[2])


def test_warp_multiband_to_grid_keeps_all_bands(tmp_path: Path) -> None:
    from eo_inference_toolbox.eocore.grid import GridSpec

    # A 3-band source (like a true-colour TCI) written on a WGS84 grid.
    cube = np.arange(3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8)
    src = str(tmp_path / "tci.tif")
    _write_cube(src, cube)
    # Warp onto a grid identical to the source footprint (x 100..108, y 44..50,
    # 1-deg pixels), so an aligned, same-resolution warp is an identity: every
    # dest pixel centre lands on a source pixel centre. The point is that all
    # three bands survive one warp (not just the first, as per-band stacking).
    grid = GridSpec(min_x=100.0, max_y=50.0, pixel_size=1.0, width=8, height=6)
    out = _raster.warp_multiband_to_grid(src, grid, _wgs84_wkt())
    assert out.shape == (3, 6, 8)
    assert np.allclose(out, cube, atol=1e-4)


def test_sample_input_max_ignores_nodata(tmp_path: Path) -> None:
    cube = np.full((1, 4, 4), 1000.0, dtype=np.float32)
    cube[0, 0, 0] = 9999.0  # this is the nodata sentinel, must be excluded
    path = str(tmp_path / "max.tif")
    _write_cube(path, cube, nodata=9999.0)
    assert _raster.sample_input_max(path, nodata=9999.0) == pytest.approx(1000.0)


def test_sample_input_max_all_nodata_returns_zero(tmp_path: Path) -> None:
    cube = np.full((1, 4, 4), -1.0, dtype=np.float32)
    path = str(tmp_path / "allnd.tif")
    _write_cube(path, cube, nodata=-1.0)
    assert _raster.sample_input_max(path, nodata=-1.0) == 0.0


def test_remove_partial_output_deletes_file_and_sidecar(tmp_path: Path) -> None:
    main = tmp_path / "partial.tif"
    sidecar = tmp_path / "partial.tif.aux.xml"
    main.write_bytes(b"partial")
    sidecar.write_text("<PAMDataset/>")
    _raster._remove_partial_output(str(main))
    assert not main.exists()
    assert not sidecar.exists()
    # A second call (nothing left to remove) is a silent no-op, never raising.
    _raster._remove_partial_output(str(main))


def test_write_geotiff_removes_partial_output_on_failure(tmp_path: Path) -> None:
    cube = np.zeros((1, 4, 4), dtype=np.float32)
    path = str(tmp_path / "broken.tif")
    # A 2-element geotransform is invalid (GDAL's SetGeoTransform typemap wants six);
    # Create() has already put a header on disk by the time it raises, so the cleanup
    # path must remove the partial file rather than leave a truncated raster behind.
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        _raster.write_geotiff(path, cube, (0.0, 1.0), _wgs84_wkt())
    assert not Path(path).exists()
