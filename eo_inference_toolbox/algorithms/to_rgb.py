"""Render a high-dimensional embedding raster to a 3-band RGB preview GeoTIFF.

Two methods: a 3-component PCA fitted on the valid pixels (default), or three
chosen embedding bands. Each channel is percentile-stretched to 8-bit; invalid
pixels are written as black.
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
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

from ..eocore.descriptor import DEFAULT_NODATA
from ..eocore.ml import valid_mask_from_nodata
from ..eocore.rgb import METHODS, apply_rgb_transform, embedding_to_rgb, fit_rgb_transform
from ..eocore.streaming import (
    DEFAULT_STREAM_TILE,
    collect_valid_sample,
    map_windows,
    plan_windows,
)
from . import _raster
from ._common import (
    EoInferenceAlgorithm,
    add_auto_low_memory_params,
    as_advanced,
    parse_band_list,
    resolve_low_memory,
)

# Cap on valid pixels sampled to fit the streamed preview transform; matches the
# PCA subsample ceiling used by the whole-cube path.
_STREAM_FIT_SAMPLES = 100_000


class ToRgbAlgorithm(EoInferenceAlgorithm):
    """Make an RGB preview of an embedding cube."""

    INPUT = "INPUT"
    METHOD = "METHOD"
    BANDS = "BANDS"
    LOW_PCT = "LOW_PCT"
    HIGH_PCT = "HIGH_PCT"
    SEED = "SEED"
    LOW_MEMORY = "LOW_MEMORY"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "torgb"

    def displayName(self) -> str:
        return "Embedding to RGB"

    def shortHelpString(self) -> str:
        return (
            "Render an embedding raster to a 3-band RGB preview, either by a "
            "3-component PCA of the valid pixels or by selecting three bands. Useful "
            "for a quick visual sanity check of an embedding."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster"))
        self.addParameter(
            QgsProcessingParameterEnum(self.METHOD, "Method", options=list(METHODS), defaultValue=0)
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BANDS,
                "Band triplet for the 'bands' method (e.g. '0 1 2')",
                optional=True,
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.LOW_PCT,
                    "Low stretch percentile",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=2.0,
                    minValue=0.0,
                    maxValue=50.0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.HIGH_PCT,
                    "High stretch percentile",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=98.0,
                    minValue=50.0,
                    maxValue=100.0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.SEED,
                    "PCA sampling seed",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=0,
                    minValue=0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.LOW_MEMORY,
                    "Low-memory mode (stream the raster in tiles instead of loading it whole)",
                    defaultValue=False,
                )
            )
        )
        # The streamed preview fits its PCA and percentile-clip bounds on a pixel sample
        # rather than every pixel, so it is not byte-identical to the whole-cube render.
        # Auto is therefore off by default: streaming only kicks in when the user opts in.
        add_auto_low_memory_params(self, auto_default=False)
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "RGB preview"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("Could not load the input embedding raster.")
        path = layer.source()
        method = METHODS[self.parameterAsEnum(parameters, self.METHOD, context)]
        triplet = self._resolve_triplet(parameters, context, method)
        low = self.parameterAsDouble(parameters, self.LOW_PCT, context)
        high = self.parameterAsDouble(parameters, self.HIGH_PCT, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)

        low_memory = resolve_low_memory(
            self,
            parameters,
            context,
            feedback,
            width=layer.width(),
            height=layer.height(),
            n_bands=layer.bandCount(),
        )
        feedback.pushInfo(f"Rendering RGB via '{method}'{' (streamed)' if low_memory else ''}...")
        if low_memory:
            result = self._torgb_streaming(path, method, triplet, low, high, seed, feedback)
        else:
            result = self._torgb_whole(path, method, triplet, low, high, seed)
        if result is None or feedback.isCanceled():
            return {}
        rgb, geotransform, wkt = result

        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            rgb,
            geotransform,
            wkt,
            band_names=["red", "green", "blue"],
            color_interp_rgb=True,
        )
        return {self.OUTPUT: output}

    def _torgb_whole(
        self,
        path: str,
        method: str,
        triplet: tuple[int, int, int] | None,
        low: float,
        high: float,
        seed: int,
    ) -> tuple[Any, Any, str] | None:
        """Default path: load the whole cube and render it in one shot."""
        raster = _raster.read_raster(path)
        nodata = raster.nodata if raster.nodata is not None else DEFAULT_NODATA
        rgb = embedding_to_rgb(
            raster.cube,
            method=method,
            bands=triplet,
            nodata=float(nodata),
            low_pct=low,
            high_pct=high,
            seed=seed,
        )
        return rgb, raster.geotransform, raster.wkt

    def _torgb_streaming(
        self,
        path: str,
        method: str,
        triplet: tuple[int, int, int] | None,
        low: float,
        high: float,
        seed: int,
        feedback: QgsProcessingFeedback,
    ) -> tuple[Any, Any, str] | None:
        """Low-memory path: fit the transform on a bounded pixel sample, apply per tile.

        Unlike the per-pixel analytics, this is *not* byte-identical to the whole-cube
        path: the PCA and the percentile clip bounds are fitted on a random sample of
        valid pixels (capped at ``_STREAM_FIT_SAMPLES``) rather than every pixel, so the
        preview can differ slightly. The whole-cube path already subsamples the PCA to
        the same ceiling, so for large rasters the two are effectively the same.
        """
        info = _raster.read_raster_info(path)
        nodata = float(info.nodata if info.nodata is not None else DEFAULT_NODATA)
        windows = plan_windows(info.width, info.height, DEFAULT_STREAM_TILE)

        def valid_fn(tile: Any) -> Any:
            return valid_mask_from_nodata(tile, nodata)

        with _raster.open_window_reader(path) as read:
            sample, n_valid = collect_valid_sample(
                windows, read, valid_fn, max_samples=_STREAM_FIT_SAMPLES, seed=seed
            )
            if sample.shape[0] == 0:
                raise QgsProcessingException("The embedding raster has no valid pixels.")
            feedback.pushInfo(
                f"Fitting RGB transform on {sample.shape[0]} of {n_valid} sampled pixel(s)."
            )
            transform = fit_rgb_transform(
                sample,
                method=method,
                bands=triplet,
                n_bands=info.n_bands,
                low_pct=low,
                high_pct=high,
                seed=seed,
            )

            def block_fn(tile: Any) -> Any:
                return apply_rgb_transform(tile, transform, nodata=nodata)

            rgb = map_windows(
                windows,
                read,
                block_fn,
                out_bands=3,
                height=info.height,
                width=info.width,
                dtype=np.uint8,
                fill=0,
                progress=feedback.setProgress,
                is_canceled=feedback.isCanceled,
            )
        if rgb is None:
            return None
        return rgb, info.geotransform, info.wkt

    def _resolve_triplet(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        method: str,
    ) -> tuple[int, int, int] | None:
        parsed = parse_band_list(self.parameterAsString(parameters, self.BANDS, context))
        if parsed is None:
            if method == "bands":
                raise QgsProcessingException(
                    "The 'bands' method needs a triplet of band indices, e.g. '0 1 2'."
                )
            return None
        try:
            indices = [int(token) for token in parsed]
        except ValueError as exc:
            raise QgsProcessingException("Band triplet must be integers.") from exc
        if len(indices) != 3:
            raise QgsProcessingException("Band triplet needs exactly three indices.")
        return (indices[0], indices[1], indices[2])
