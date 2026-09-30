"""Fetch analysis-ready satellite imagery for an AOI from a public STAC API.

Picks the least-cloudy scene in the date range for the chosen backend's default
collection (Sentinel-2 L2A on Earth Search, or HLS on the Planetary Computer),
warps its bands onto the target grid, and writes a multi-band GeoTIFF stamped
with the semantic band names and ground sample distance the ``embed`` algorithm
needs to build backend metadata.

When the backend declares a pre-rendered ``fetch_asset`` (SatlasPretrain uses
Sentinel-2's 8-bit ``visual`` / TCI true-colour COG), that single multi-band asset
is warped in one pass instead of resolving one reflectance COG per band, so the
fetched imagery is already in the 0-255 range the RGB checkpoint expects and the
Embed step needs no manual "Input scale".
"""

from __future__ import annotations

from typing import Any

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterString,
)

from ..eocore import stac
from ..eocore.backends import backend_info
from ..eocore.composite import COMPOSITE_MODES, composite
from ..eocore.grid import metres_to_degrees
from . import _raster, _stac_fetch
from ._common import (
    BACKEND_LABELS,
    MD_BANDS,
    MD_COLLECTION,
    MD_GSD,
    EoInferenceAlgorithm,
    as_advanced,
    backend_id_from_index,
    grid_from_extent,
)


class FetchImageryAlgorithm(EoInferenceAlgorithm):
    """Download and mosaic source imagery for an AOI."""

    EXTENT = "EXTENT"
    CRS = "CRS"
    BACKEND = "BACKEND"
    START = "START"
    END = "END"
    MAX_CLOUD = "MAX_CLOUD"
    PIXEL_SIZE = "PIXEL_SIZE"
    COMPOSITE = "COMPOSITE"
    COMPOSITE_MODE = "COMPOSITE_MODE"
    MAX_SCENES = "MAX_SCENES"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "fetchimagery"

    def displayName(self) -> str:
        return "Fetch imagery (STAC)"

    def shortHelpString(self) -> str:
        return (
            "Fetch the least-cloudy scene over the extent for the chosen backend's "
            "default collection and write it as a multi-band GeoTIFF.\n\n"
            "Sentinel-2 L2A is read anonymously from Earth Search. HLS is read from "
            "the Microsoft Planetary Computer and its assets are signed automatically "
            "when the 'planetary_computer' package is installed.\n\n"
            "For SatlasPretrain the pre-rendered 8-bit true-colour asset (Sentinel-2's "
            "'visual' / TCI: red/green/blue, 0-255) is fetched instead of raw reflectance "
            "bands, so the imagery matches the RGB checkpoint and the Embed step needs no "
            "manual 'Input scale'."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, "Area of interest"))
        self.addParameter(
            QgsProcessingParameterCrs(
                self.CRS, "Output CRS", defaultValue="ProjectCrs", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.BACKEND, "Backend", options=BACKEND_LABELS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterString(self.START, "Start date (YYYY-MM-DD)", optional=True)
        )
        self.addParameter(
            QgsProcessingParameterString(self.END, "End date (YYYY-MM-DD)", optional=True)
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_CLOUD,
                "Maximum cloud cover (%)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=20.0,
                minValue=0.0,
                maxValue=100.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PIXEL_SIZE,
                "Pixel size (CRS units; 0 = backend native)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.COMPOSITE,
                    "Composite several scenes (reduce cloud/gaps) instead of one",
                    defaultValue=False,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterEnum(
                    self.COMPOSITE_MODE,
                    "Composite reduction",
                    options=list(COMPOSITE_MODES),
                    defaultValue=list(COMPOSITE_MODES).index("median"),
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.MAX_SCENES,
                    "Composite: maximum scenes to combine",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=6,
                    minValue=2,
                )
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Imagery"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        backend_id = backend_id_from_index(self.parameterAsEnum(parameters, self.BACKEND, context))
        info = backend_info(backend_id)
        collection = info.default_collection
        bands = info.default_bands
        fetch_asset = info.fetch_asset

        crs = self._resolve_crs(self.parameterAsCrs(parameters, self.CRS, context))
        pixel_size = self.parameterAsDouble(parameters, self.PIXEL_SIZE, context)
        if pixel_size <= 0:
            # No explicit pixel size: fall back to the backend's native resolution.
            # The grid pixel is expressed in the target CRS's units, so in a
            # geographic CRS convert the metre resolution to a degree-equivalent --
            # using the metre count verbatim would build a grossly oversized grid
            # (e.g. a 10 m default becoming 10 degrees per pixel).
            pixel_size = (
                metres_to_degrees(info.resolution_m) if crs.isGeographic() else info.resolution_m
            )
        rect = self.parameterAsExtent(parameters, self.EXTENT, context, crs)
        grid = grid_from_extent(rect, pixel_size)

        wgs = self.parameterAsExtent(
            parameters, self.EXTENT, context, QgsCoordinateReferenceSystem("EPSG:4326")
        )
        bbox = (wgs.xMinimum(), wgs.yMinimum(), wgs.xMaximum(), wgs.yMaximum())

        start = self.parameterAsString(parameters, self.START, context) or None
        end = self.parameterAsString(parameters, self.END, context) or None
        max_cloud = self.parameterAsDouble(parameters, self.MAX_CLOUD, context)

        items = self._search(collection, bbox, start, end, max_cloud, feedback)
        wkt = str(crs.toWkt())
        if self.parameterAsBool(parameters, self.COMPOSITE, context):
            stack = self._composite_stack(
                parameters, context, feedback, items, bands, grid, wkt, fetch_asset
            )
        else:
            stack = self._single_stack(feedback, items[0], bands, grid, wkt, fetch_asset)
        if stack is None or feedback.isCanceled():
            return {}

        # Whichever fetch path ran, the written cube must carry exactly the bands the
        # backend metadata will claim (embed reads MD_BANDS back). The per-band path
        # stacks one plane per band by construction; this guards the pre-rendered
        # fetch_asset path against a scene whose asset has an unexpected band count.
        if int(stack.shape[0]) != len(bands):
            raise QgsProcessingException(
                f"Fetched imagery has {int(stack.shape[0])} band(s) but the "
                f"{info.display_name} backend expects {len(bands)} ({', '.join(bands)}). "
                "The fetched STAC asset does not match the backend's band convention."
            )

        gsd = pixel_size if not crs.isGeographic() else info.resolution_m
        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            stack,
            grid.geotransform,
            str(crs.toWkt()),
            band_names=list(bands),
            metadata={
                MD_BANDS: ",".join(bands),
                MD_GSD: f"{gsd:g}",
                MD_COLLECTION: collection,
            },
        )
        return {self.OUTPUT: output}

    def _search(
        self,
        collection: str,
        bbox: tuple[float, float, float, float],
        start: str | None,
        end: str | None,
        max_cloud: float,
        feedback: QgsProcessingFeedback,
    ) -> list[stac.StacItem]:
        """Return matching scenes sorted least-cloudy first (unknown cover last)."""
        if collection == stac.S2_L2A:
            spec = stac.sentinel2_search(bbox, start=start, end=end, max_cloud=max_cloud)
        else:
            sensor = "L30" if collection == stac.HLS_L30 else "S30"
            spec = stac.hls_search(bbox, sensor=sensor, start=start, end=end, max_cloud=max_cloud)
        feedback.pushInfo(f"Searching {collection} over {bbox}...")
        try:
            items = stac.search(spec)
        except OSError as exc:
            raise QgsProcessingException(f"STAC search failed: {exc}") from exc
        if not items:
            raise QgsProcessingException(
                "No scenes found for that extent, date range and cloud limit."
            )
        items.sort(key=lambda it: it.cloud_cover if it.cloud_cover is not None else float("inf"))
        return items

    def _warp_item(
        self,
        feedback: QgsProcessingFeedback,
        item: stac.StacItem,
        bands: tuple[str, ...],
        grid: Any,
        wkt: str,
        fetch_asset: str | None = None,
    ) -> Any:
        """Resolve + sign the source(s) for one item and warp onto the grid.

        With ``fetch_asset`` set (e.g. Satlas' ``"visual"``) a single pre-rendered
        multi-band asset is warped in one pass; otherwise one single-band COG per
        semantic band is resolved through the collection's band plan and stacked.
        """
        if fetch_asset:
            hrefs = [stac.href_for_asset(item, fetch_asset)]
        else:
            hrefs = stac.hrefs_for_bands(item, bands)
        signed, missing_signer = _stac_fetch.sign_hrefs(hrefs)
        if missing_signer:
            feedback.pushWarning(
                "HLS assets need signing but the 'planetary_computer' package is not "
                "installed; the read may fail. Install it with pip and retry."
            )
        if fetch_asset:
            return _raster.warp_multiband_to_grid(signed[0], grid, wkt)
        return _raster.warp_bands_to_grid(signed, grid, wkt)

    def _single_stack(
        self,
        feedback: QgsProcessingFeedback,
        item: stac.StacItem,
        bands: tuple[str, ...],
        grid: Any,
        wkt: str,
        fetch_asset: str | None = None,
    ) -> Any:
        cloud = "unknown" if item.cloud_cover is None else f"{item.cloud_cover:.1f}%"
        feedback.pushInfo(f"Selected scene {item.id} (cloud cover {cloud}).")
        if fetch_asset:
            feedback.pushInfo(
                f"Warping the pre-rendered '{fetch_asset}' asset onto the target grid..."
            )
        else:
            feedback.pushInfo(f"Warping {len(bands)} band(s) onto the target grid...")
        return self._warp_item(feedback, item, bands, grid, wkt, fetch_asset)

    def _composite_stack(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        items: list[stac.StacItem],
        bands: tuple[str, ...],
        grid: Any,
        wkt: str,
        fetch_asset: str | None = None,
    ) -> Any:
        max_scenes = self.parameterAsInt(parameters, self.MAX_SCENES, context)
        mode = list(COMPOSITE_MODES)[self.parameterAsEnum(parameters, self.COMPOSITE_MODE, context)]
        chosen = items[:max_scenes]
        feedback.pushInfo(f"Compositing {len(chosen)} scene(s) with '{mode}' reduction...")
        cubes = []
        for index, item in enumerate(chosen, start=1):
            if feedback.isCanceled():
                return None
            feedback.pushInfo(f"  scene {index}/{len(chosen)}: {item.id}")
            cubes.append(self._warp_item(feedback, item, bands, grid, wkt, fetch_asset))
        return composite(cubes, mode=mode)
