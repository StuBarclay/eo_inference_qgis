"""Run an inference backend over an imagery raster to produce an embedding cube.

The scene is cut into overlapping tiles (:mod:`eocore.tiling`), each tile is
embedded by the chosen backend, and the per-tile outputs are stitched with
overlap averaging. The mosaic is then L2-normalised (all three backends emit
non-unit vectors) so downstream similarity/change can use the dot-product
shortcut, and the descriptor is stamped into the GeoTIFF metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

from ..eocore.backends import (
    TileMeta,
    backend_info,
    band_requirement_error,
    embed_tile_batch,
    load_backend,
    scale_sanity_warning,
)
from ..eocore.backends._pool import UPSAMPLING_METHODS
from ..eocore.backends._weights import human_bytes
from ..eocore.descriptor import DEFAULT_NODATA
from ..eocore.runner import finalize_embedding, run_windowed_embed
from ..eocore.tiling import MosaicAccumulator, cosine_taper_weights, iter_windows
from . import _raster
from ._common import (
    BACKEND_LABELS,
    MD_BANDS,
    EoInferenceAlgorithm,
    as_advanced,
    backend_id_from_index,
    parse_band_list,
    read_gsd,
)


class EmbedAlgorithm(EoInferenceAlgorithm):
    """Embed an imagery raster with a foundation-model backend."""

    INPUT = "INPUT"
    BACKEND = "BACKEND"
    BANDS = "BANDS"
    TILE = "TILE"
    OVERLAP = "OVERLAP"
    GSD = "GSD"
    INPUT_SCALE = "INPUT_SCALE"
    UPSAMPLING = "UPSAMPLING"
    BATCH_SIZE = "BATCH_SIZE"
    FEATHER = "FEATHER"
    DETECT_VOID = "DETECT_VOID"
    WEIGHTS = "WEIGHTS"
    CACHE_DIR = "CACHE_DIR"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "embed"

    def displayName(self) -> str:
        return "Embed imagery"

    def shortHelpString(self) -> str:
        return (
            "Run a foundation-model backend (Clay, Prithvi or Satlas) over an imagery "
            "raster and write a per-pixel embedding GeoTIFF. The output is "
            "L2-normalised and tagged with the backend id so similarity, change, "
            "classify and cluster stay correct.\n\n"
            "PyTorch and the backend's model package must be installed (the "
            "'inference' dependency tier); weights download on first use."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Imagery raster"))
        self.addParameter(
            QgsProcessingParameterEnum(
                self.BACKEND, "Backend", options=BACKEND_LABELS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BANDS,
                "Band order override (e.g. 'red green blue nir'; blank = auto)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TILE,
                "Tile size (pixels)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=256,
                minValue=16,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OVERLAP,
                "Tile overlap (pixels)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=32,
                minValue=0,
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.GSD,
                    "Ground sample distance override (m; 0 = auto)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=0.0,
                    minValue=0.0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.INPUT_SCALE,
                    (
                        "Input scale multiplier (0 = backend default). Clay/Prithvi "
                        "standardise raw DN internally, so leave 0 for DN imagery and "
                        "set 1e4 only if yours is already reflectance in [0,1]; Satlas "
                        "expects 8-bit true colour (default 1/255)"
                    ),
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=0.0,
                    minValue=0.0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterEnum(
                    self.UPSAMPLING,
                    "Patch-grid upsampling (bilinear is smoother but blends patch edges)",
                    options=list(UPSAMPLING_METHODS),
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.BATCH_SIZE,
                    "Tiles per forward pass (1 = one tile at a time; raise for GPU throughput)",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=1,
                    minValue=1,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.FEATHER,
                    "Overlap feathering taper (pixels; 0 = uniform blend, off)",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=0,
                    minValue=0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.DETECT_VOID,
                    (
                        "Treat all-zero (black) pixels as off-scene void "
                        "(rotated / reprojected scenes; declared nodata is always honoured)"
                    ),
                    defaultValue=False,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterFile(
                    self.WEIGHTS,
                    "Local weights checkpoint (blank = download)",
                    optional=True,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterFile(
                    self.CACHE_DIR,
                    "Weights cache folder (blank = default)",
                    behavior=QgsProcessingParameterFile.Behavior.Folder,
                    optional=True,
                )
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Embedding"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("Could not load the input imagery raster.")
        path = layer.source()
        backend_id = backend_id_from_index(self.parameterAsEnum(parameters, self.BACKEND, context))
        info = backend_info(backend_id)

        bands = self._resolve_bands(parameters, context, path, info.default_bands)
        raster_info = _raster.read_raster_info(path)
        n_in = raster_info.n_bands
        if n_in != len(bands):
            raise QgsProcessingException(
                f"The raster has {n_in} band(s) but the backend expects "
                f"{len(bands)} ({', '.join(bands)}). Set the band-order override or "
                "fetch imagery with this backend first."
            )
        # A strict backend (Prithvi, Satlas) needs an exact band set; the count
        # check above passes when the imagery's stamped bands merely happen to match
        # the resolved count, so catch a wrong-backend-for-this-imagery mismatch here
        # with a clear, backend-aware message instead of the model's opaque "got N".
        mismatch = band_requirement_error(backend_id, bands)
        if mismatch is not None:
            raise QgsProcessingException(mismatch)
        self._warn_on_scale(parameters, context, info, path, raster_info.nodata, feedback)
        grid = raster_info.grid_spec()
        lon, lat = _raster.grid_center_lonlat(grid, raster_info.wkt)
        gsd_param = self.parameterAsDouble(parameters, self.GSD, context)
        gsd = gsd_param if gsd_param > 0 else read_gsd(path, grid.pixel_size)

        tile = self.parameterAsInt(parameters, self.TILE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        if overlap >= tile:
            raise QgsProcessingException("Overlap must be smaller than the tile size.")

        backend = self._load_backend(parameters, context, backend_id, feedback)
        meta = TileMeta(bands=tuple(bands), gsd=float(gsd), lon=lon, lat=lat)
        batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        taper = self.parameterAsInt(parameters, self.FEATHER, context)
        detect_void = self.parameterAsBool(parameters, self.DETECT_VOID, context)
        mosaic = self._run_tiles(
            path,
            grid,
            backend,
            meta,
            tile,
            overlap,
            batch_size,
            taper,
            raster_info.nodata,
            detect_void,
            feedback,
        )
        if mosaic is None:
            return {}

        normalized, _coverage, descriptor = finalize_embedding(mosaic, backend.descriptor())
        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            normalized,
            grid.geotransform,
            raster_info.wkt,
            nodata=DEFAULT_NODATA,
            metadata=self._stamp(descriptor),
        )
        feedback.pushInfo(
            f"Wrote a {normalized.shape[0]}-band embedding from backend '{backend_id}'."
        )
        return {self.OUTPUT: output}

    # -- helpers -------------------------------------------------------------
    def _resolve_bands(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        path: str,
        default_bands: tuple[str, ...],
    ) -> tuple[str, ...]:
        override = parse_band_list(self.parameterAsString(parameters, self.BANDS, context))
        if override is not None:
            return override
        stamped = _raster.read_metadata_item(path, MD_BANDS)
        from_meta = parse_band_list(stamped)
        if from_meta is not None:
            return from_meta
        return tuple(default_bands)

    def _warn_on_scale(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        info: Any,
        path: str,
        nodata: float | None,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """Push a feedback warning if the imagery range looks mis-scaled for the model.

        The maximum is taken from a bounded, decimated read so this check never
        loads the full cube (the windowed embed loop reads tile-by-tile).
        """
        user_scale = self.parameterAsDouble(parameters, self.INPUT_SCALE, context)
        effective = user_scale if user_scale > 0 else info.default_input_scale
        message = scale_sanity_warning(_raster.sample_input_max(path, nodata), effective)
        if message:
            feedback.pushWarning(message)

    def _download_progress(self, feedback: QgsProcessingFeedback) -> Any:
        """A weight-download progress callback that logs through ``feedback`` (E-5).

        Announces the first-run download once, then reports at ~10% steps when a
        ``Content-Length`` is known. It uses ``pushInfo`` (not ``setProgress``) so
        it never fights the tiling progress bar that runs afterwards.
        """
        state = {"announced": False, "last_pct": -10}

        def report(done: int, total: int | None) -> None:
            if not state["announced"]:
                feedback.pushInfo(
                    "Preparing model weights (downloaded and cached on first run; "
                    "this can take several minutes)..."
                )
                state["announced"] = True
            if total and total > 0:
                pct = int(done * 100 / total)
                if pct >= int(state["last_pct"]) + 10 or pct >= 100:
                    state["last_pct"] = pct
                    feedback.pushInfo(
                        f"  weights: {pct}% ({human_bytes(done)} / {human_bytes(total)})"
                    )

        return report

    def _load_backend(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        backend_id: str,
        feedback: QgsProcessingFeedback,
    ) -> Any:
        weights = self.parameterAsFile(parameters, self.WEIGHTS, context) or None
        cache_dir = self.parameterAsFile(parameters, self.CACHE_DIR, context) or None
        config: dict[str, Any] = {"progress": self._download_progress(feedback)}
        if weights:
            config["weights"] = weights
        if cache_dir:
            config["cache_dir"] = cache_dir
        input_scale = self.parameterAsDouble(parameters, self.INPUT_SCALE, context)
        if input_scale > 0:
            # Satlas names its knob 'input_scale'; the ViT backends 'reflectance_scale'.
            key = "input_scale" if backend_id == "satlas" else "reflectance_scale"
            config[key] = input_scale
        method_index = self.parameterAsEnum(parameters, self.UPSAMPLING, context)
        method = list(UPSAMPLING_METHODS)[method_index]
        if method != "nearest":
            config["upsampling"] = method
        try:
            return load_backend(backend_id, **config)
        except (ImportError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

    def _run_tiles(
        self,
        path: str,
        grid: Any,
        backend: Any,
        meta: TileMeta,
        tile: int,
        overlap: int,
        batch_size: int,
        taper: int,
        input_nodata: float | None,
        detect_void: bool,
        feedback: QgsProcessingFeedback,
    ) -> MosaicAccumulator | None:
        """Stream windows off disk, embed them in batches and mosaic the result.

        Reads one tile window at a time (E-1) so peak memory tracks the tile
        size, not the scene; batches up to ``batch_size`` windows per forward
        pass (E-2); and applies an optional cosine taper for overlap feathering
        (E-4) when ``taper > 0``. ``input_nodata`` (the raster's declared nodata,
        honoured by default) and ``detect_void`` (opt-in all-zero detection) keep
        off-scene / nodata input pixels out of the forward pass and out of the
        mosaic coverage, so they become nodata rather than a bogus embedding.
        """
        windows = list(iter_windows(grid.width, grid.height, tile, overlap))

        def embed_batch(chips: list[Any]) -> list[Any]:
            tiles = [np.asarray(chip, dtype=np.float32) for chip in chips]
            try:
                return embed_tile_batch(backend, tiles, meta)
            except (ImportError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc

        def taper_weight(window: Any) -> Any:
            return cosine_taper_weights(window.height, window.width, taper)

        weight_fn = taper_weight if taper > 0 else None

        try:
            with _raster.open_window_reader(path) as read_window:
                return run_windowed_embed(
                    windows,
                    grid.height,
                    grid.width,
                    read_window,
                    embed_batch,
                    batch_size=batch_size,
                    weight_fn=weight_fn,
                    input_nodata=input_nodata,
                    detect_void=detect_void,
                    progress=feedback.setProgress,
                    is_canceled=feedback.isCanceled,
                )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
