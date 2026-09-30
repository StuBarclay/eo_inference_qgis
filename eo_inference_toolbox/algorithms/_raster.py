"""GDAL glue: read cubes, write GeoTIFFs, warp imagery onto a target grid.

This is the only module in the toolbox that touches GDAL directly. GDAL ships
with QGIS, so it is always importable at algorithm run time; even so we import it
lazily -- inside the functions that use it, through :func:`_gdal` / :func:`_osr`,
never at module import. That keeps ``import`` of this module (and, through it, the
whole ``algorithms`` package and the Processing provider) cheap and unable to fail
when GDAL is absent -- exactly like the ``_sklearn`` and ``_stac_fetch`` optional
tiers. The pure grid maths lives in :mod:`eocore.grid`; here we only bridge to GDAL.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..eocore.grid import Bounds, GridSpec
from ..eocore.tiling import Window


def _remove_partial_output(path: str) -> None:
    """Best-effort removal of a partially written raster (and its ``.aux.xml`` sidecar).

    Called when a write/warp fails midway so a truncated GeoTIFF is not left at the
    output path for a downstream read to mistake for a complete result. Never raises:
    cleanup of a failed write must not mask the original error.
    """
    for candidate in (path, f"{path}.aux.xml"):
        with contextlib.suppress(OSError):
            if os.path.exists(candidate):
                os.remove(candidate)


def _gdal() -> Any:
    """Import GDAL lazily and enable Python exceptions (idempotent)."""
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def _osr() -> Any:
    """Import OSR lazily and enable Python exceptions (idempotent)."""
    from osgeo import osr

    osr.UseExceptions()
    return osr


# GDAL config applied while reading remote (/vsicurl) sources.
_REMOTE_CONFIG = {
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "VSI_CACHE": "TRUE",
}

# Metadata keys stamped into embedding rasters so downstream reads the descriptor.
MD_BACKEND = "EOINFER_BACKEND"
MD_UNIT = "EOINFER_UNIT_NORMALIZED"


@dataclass(frozen=True)
class RasterData:
    """A raster read into memory: cube plus georeferencing."""

    cube: NDArray[Any]
    geotransform: tuple[float, float, float, float, float, float]
    wkt: str
    nodata: float | None

    def grid_spec(self) -> GridSpec:
        gt = self.geotransform
        _, _, height, width = _cube_shape(self.cube)
        return GridSpec(
            min_x=gt[0],
            max_y=gt[3],
            pixel_size=abs(gt[1]),
            width=width,
            height=height,
        )


def _cube_shape(cube: NDArray[Any]) -> tuple[int, int, int, int]:
    if cube.ndim == 2:
        return (1, 1, cube.shape[0], cube.shape[1])
    if cube.ndim == 3:
        bands, height, width = cube.shape
        return (bands, bands, height, width)
    raise ValueError(f"expected a 2-D or 3-D array, got shape {cube.shape}")


@contextlib.contextmanager
def remote_config() -> Iterator[None]:
    """Apply the /vsicurl-friendly GDAL config for the duration of a read."""
    gdal = _gdal()
    previous: dict[str, str | None] = {}
    for key, value in _REMOTE_CONFIG.items():
        previous[key] = gdal.GetConfigOption(key)
        gdal.SetConfigOption(key, value)
    try:
        yield
    finally:
        for key, old in previous.items():
            gdal.SetConfigOption(key, old)


def vsicurl(url: str) -> str:
    """Wrap an HTTP(S) URL as a GDAL ``/vsicurl`` path (idempotent)."""
    if url.startswith(("/vsicurl/", "/vsi")):
        return url
    if url.startswith(("http://", "https://")):
        return "/vsicurl/" + url
    return url


def read_raster(path: str, *, bands: Sequence[int] | None = None) -> RasterData:
    """Read a raster into a ``(bands, H, W)`` cube (1-based ``bands`` optional)."""
    gdal = _gdal()
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"could not open raster: {path!r}")
    geotransform = dataset.GetGeoTransform()
    wkt = dataset.GetProjection()
    first_band = dataset.GetRasterBand(1)
    nodata = first_band.GetNoDataValue()
    if bands is None:
        array = dataset.ReadAsArray()
    else:
        array = np.stack([dataset.GetRasterBand(int(b)).ReadAsArray() for b in bands], axis=0)
    cube = np.asarray(array)
    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]
    return RasterData(cube=cube, geotransform=geotransform, wkt=wkt, nodata=nodata)


def read_metadata_item(path: str, key: str) -> str | None:
    """Return a dataset-level metadata item, or ``None`` if absent."""
    gdal = _gdal()
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        return None
    value = dataset.GetMetadataItem(key)
    return str(value) if value is not None else None


@dataclass(frozen=True)
class RasterInfo:
    """Georeferencing + shape read from a raster header, without its pixels.

    Lets the ``embed`` algorithm set up its grid, band count and scale check up
    front while reading pixels one window at a time (E-1 / H-5), rather than
    pulling the whole cube into RAM via :func:`read_raster`.
    """

    geotransform: tuple[float, float, float, float, float, float]
    wkt: str
    nodata: float | None
    n_bands: int
    width: int
    height: int

    def grid_spec(self) -> GridSpec:
        gt = self.geotransform
        return GridSpec(
            min_x=gt[0],
            max_y=gt[3],
            pixel_size=abs(gt[1]),
            width=self.width,
            height=self.height,
        )


def read_raster_info(path: str) -> RasterInfo:
    """Read georeferencing and shape from a raster's header (no pixel read)."""
    gdal = _gdal()
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"could not open raster: {path!r}")
    info = RasterInfo(
        geotransform=dataset.GetGeoTransform(),
        wkt=dataset.GetProjection(),
        nodata=dataset.GetRasterBand(1).GetNoDataValue(),
        n_bands=int(dataset.RasterCount),
        width=int(dataset.RasterXSize),
        height=int(dataset.RasterYSize),
    )
    dataset = None
    return info


@contextlib.contextmanager
def open_window_reader(
    path: str, *, bands: Sequence[int] | None = None
) -> Iterator[Callable[[Window], NDArray[Any]]]:
    """Open a raster once and yield a callable that reads one window at a time.

    Each call issues a windowed ``ReadAsArray`` and returns a ``(bands, h, w)``
    array — only the requested block is materialised, so peak memory tracks the
    tile size, not the whole scene (E-1 / H-5). The dataset stays open for the
    lifetime of the context so successive window reads reuse GDAL's block cache.
    """
    gdal = _gdal()
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"could not open raster: {path!r}")
    band_list = [int(b) for b in bands] if bands is not None else None

    def read(window: Window) -> NDArray[Any]:
        col_off, row_off = int(window.col_off), int(window.row_off)
        win_w, win_h = int(window.width), int(window.height)
        if band_list is None:
            array = dataset.ReadAsArray(col_off, row_off, win_w, win_h)
        else:
            planes = [
                dataset.GetRasterBand(b).ReadAsArray(col_off, row_off, win_w, win_h)
                for b in band_list
            ]
            array = np.stack(planes, axis=0)
        cube = np.asarray(array)
        if cube.ndim == 2:
            cube = cube[np.newaxis, :, :]
        return cube

    try:
        yield read
    finally:
        dataset = None


def sample_input_max(path: str, nodata: float | None = None, *, max_dim: int = 1024) -> float:
    """Largest finite, non-nodata pixel value from a decimated read of ``path``.

    Reads the raster downsampled so its longest side is at most ``max_dim``
    pixels — a bounded, memory-safe scan used only for the input-scale sanity
    check (H-7), so the embed path need never load the full cube (E-1 / H-5).
    Returns ``0.0`` when every sampled pixel is nodata / non-finite.
    """
    gdal = _gdal()
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"could not open raster: {path!r}")
    full_w, full_h = int(dataset.RasterXSize), int(dataset.RasterYSize)
    step = max(1, math.ceil(max(full_w, full_h) / max(1, max_dim)))
    buf_w, buf_h = max(1, full_w // step), max(1, full_h // step)
    array = np.asarray(dataset.ReadAsArray(buf_xsize=buf_w, buf_ysize=buf_h))
    dataset = None
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    mask = np.isfinite(array)
    if nodata is not None:
        mask &= array != nodata
    if not bool(mask.any()):
        return 0.0
    return float(array[mask].max())


def write_geotiff(
    path: str,
    cube: NDArray[Any],
    geotransform: Sequence[float],
    wkt: str,
    *,
    nodata: float | None = None,
    band_names: Sequence[str] | None = None,
    metadata: dict[str, str] | None = None,
    color_interp_rgb: bool = False,
) -> None:
    """Write a ``(bands, H, W)`` cube to a GeoTIFF."""
    from osgeo import gdal_array

    gdal = _gdal()
    arr = np.asarray(cube)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    n_bands, height, width = arr.shape
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(arr.dtype)
    if gdal_dtype is None:
        arr = arr.astype(np.float32)
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(np.float32)
    driver = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    dataset = driver.Create(path, width, height, n_bands, gdal_dtype, options)
    if dataset is None:
        raise RuntimeError(f"could not create raster: {path!r}")
    try:
        dataset.SetGeoTransform(tuple(float(v) for v in geotransform))
        dataset.SetProjection(wkt)
        if metadata:
            dataset.SetMetadata({k: str(v) for k, v in metadata.items()})
        rgb_interp = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand]
        for index in range(n_bands):
            band = dataset.GetRasterBand(index + 1)
            band.WriteArray(arr[index])
            if nodata is not None:
                band.SetNoDataValue(float(nodata))
            if band_names is not None and index < len(band_names):
                band.SetDescription(str(band_names[index]))
            if color_interp_rgb and index < 3:
                band.SetColorInterpretation(rgb_interp[index])
        dataset.FlushCache()
    except Exception:
        # Release the handle first (Windows won't delete an open file), then remove
        # the partial output before re-raising so no truncated GeoTIFF is left behind.
        dataset = None
        _remove_partial_output(path)
        raise
    dataset = None


def bounds_to_wgs84(bounds: Bounds, src_wkt: str) -> Bounds:
    """Reproject an axis-aligned bounding box to WGS84 (lon/lat) corners."""
    osr = _osr()
    source = osr.SpatialReference()
    source.ImportFromWkt(src_wkt)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    min_x, min_y, max_x, max_y = bounds
    corners = [
        transform.TransformPoint(min_x, min_y),
        transform.TransformPoint(min_x, max_y),
        transform.TransformPoint(max_x, min_y),
        transform.TransformPoint(max_x, max_y),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def grid_center_lonlat(grid: GridSpec, wkt: str) -> tuple[float, float]:
    """Return the (lon, lat) of a grid's centre for backend metadata."""
    osr = _osr()
    min_x, min_y, max_x, max_y = grid.bounds
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    source = osr.SpatialReference()
    source.ImportFromWkt(wkt)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    lon, lat, _ = transform.TransformPoint(center_x, center_y)
    return (float(lon), float(lat))


def warp_bands_to_grid(
    hrefs: Sequence[str],
    grid: GridSpec,
    dst_wkt: str,
    *,
    resample: str = "bilinear",
    src_nodata: float | None = None,
) -> NDArray[Any]:
    """Warp each single-band source onto ``grid`` and stack into ``(bands, H, W)``.

    Each href is treated as one band (Sentinel-2 / HLS assets are per-band COGs).
    Remote sources are read through ``/vsicurl``.
    """
    gdal = _gdal()
    if not hrefs:
        raise ValueError("need at least one source href")
    min_x, min_y, max_x, max_y = grid.bounds
    warp_options = gdal.WarpOptions(
        format="MEM",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=grid.width,
        height=grid.height,
        dstSRS=dst_wkt,
        resampleAlg=resample,
        srcNodata=src_nodata,
        multithread=True,
    )
    planes: list[NDArray[Any]] = []
    with remote_config():
        for href in hrefs:
            source = vsicurl(href)
            warped = gdal.Warp("", source, options=warp_options)
            if warped is None:
                raise RuntimeError(f"failed to warp source: {href!r}")
            planes.append(np.asarray(warped.ReadAsArray()))
            warped = None
    return np.stack(planes, axis=0)


def warp_multiband_to_grid(
    href: str,
    grid: GridSpec,
    dst_wkt: str,
    *,
    resample: str = "bilinear",
    src_nodata: float | None = None,
) -> NDArray[Any]:
    """Warp one *remote multi-band* asset onto ``grid`` and return ``(bands, H, W)``.

    The remote counterpart of :func:`warp_raster_to_grid`: it reads a single
    source through ``/vsicurl`` (under :func:`remote_config`, like
    :func:`warp_bands_to_grid`) but keeps every band of that one asset rather than
    treating each href as a separate band. Used to fetch a pre-rendered multi-band
    STAC asset in one pass — e.g. Sentinel-2's 8-bit ``visual`` (TCI) true-colour
    COG for the Satlas RGB checkpoint. Returns just the pixel stack (the
    georeferencing is fixed by ``grid``/``dst_wkt``); a single-band source is
    promoted to a leading band axis.
    """
    gdal = _gdal()
    min_x, min_y, max_x, max_y = grid.bounds
    warp_options = gdal.WarpOptions(
        format="MEM",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=grid.width,
        height=grid.height,
        dstSRS=dst_wkt,
        resampleAlg=resample,
        srcNodata=src_nodata,
        multithread=True,
    )
    with remote_config():
        warped = gdal.Warp("", vsicurl(href), options=warp_options)
        if warped is None:
            raise RuntimeError(f"failed to warp source: {href!r}")
        arr = np.asarray(warped.ReadAsArray())
        warped = None
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


def warp_raster_to_grid(
    path: str,
    grid: GridSpec,
    dst_wkt: str,
    *,
    resample: str = "bilinear",
    src_nodata: float | None = None,
) -> RasterData:
    """Warp an entire multi-band raster onto ``grid`` (reprojecting if needed).

    Unlike :func:`warp_bands_to_grid` (which treats each href as one band), this
    warps all bands of a single dataset in one pass — used to align embedding
    rasters that were written on different grids before a change computation.
    """
    gdal = _gdal()
    min_x, min_y, max_x, max_y = grid.bounds
    warp_options = gdal.WarpOptions(
        format="MEM",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=grid.width,
        height=grid.height,
        dstSRS=dst_wkt,
        resampleAlg=resample,
        srcNodata=src_nodata,
        dstNodata=src_nodata,
        multithread=True,
    )
    warped = gdal.Warp("", path, options=warp_options)
    if warped is None:
        raise RuntimeError(f"failed to warp raster: {path!r}")
    arr = np.asarray(warped.ReadAsArray())
    geotransform = warped.GetGeoTransform()
    wkt = warped.GetProjection()
    nodata = warped.GetRasterBand(1).GetNoDataValue()
    warped = None
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return RasterData(cube=arr, geotransform=geotransform, wkt=wkt, nodata=nodata)


def warp_raster_to_grid_file(
    path: str,
    grid: GridSpec,
    dst_wkt: str,
    out_path: str,
    *,
    resample: str = "nearest",
    src_nodata: float | None = None,
) -> None:
    """Warp a whole multi-band raster onto ``grid`` and write it to ``out_path``.

    The on-disk twin of :func:`warp_raster_to_grid`: GDAL streams the warp straight
    to a tiled, compressed GeoTIFF rather than a MEM dataset, so the low-memory
    change path can align a mismatched raster without ever holding a full warped
    cube in RAM, then stream the result back one window at a time.
    """
    gdal = _gdal()
    min_x, min_y, max_x, max_y = grid.bounds
    warp_options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=grid.width,
        height=grid.height,
        dstSRS=dst_wkt,
        resampleAlg=resample,
        srcNodata=src_nodata,
        dstNodata=src_nodata,
        multithread=True,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    try:
        warped = gdal.Warp(out_path, path, options=warp_options)
    except Exception:
        # Remove any partial output GDAL streamed before it failed, then re-raise.
        _remove_partial_output(out_path)
        raise
    if warped is None:
        _remove_partial_output(out_path)
        raise RuntimeError(f"failed to warp raster: {path!r}")
    warped = None


def cubes_aligned(a: RasterData, b: RasterData, *, tol: float = 1e-6) -> bool:
    """True if two rasters share pixel grid (shape + geotransform within tol)."""
    if a.cube.shape[1:] != b.cube.shape[1:]:
        return False
    return all(abs(x - y) <= tol for x, y in zip(a.geotransform, b.geotransform, strict=True))


def infos_aligned(a: RasterInfo, b: RasterInfo, *, tol: float = 1e-6) -> bool:
    """True if two raster headers share pixel grid (shape + geotransform within tol).

    The header-only twin of :func:`cubes_aligned`, so the low-memory change path can
    decide whether a raster needs warping from :func:`read_raster_info` alone,
    without reading either cube.
    """
    if (a.height, a.width) != (b.height, b.width):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a.geotransform, b.geotransform, strict=True))
