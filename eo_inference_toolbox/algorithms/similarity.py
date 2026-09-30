"""Per-pixel cosine similarity of an embedding raster to reference points.

The reference vector is the (normalised) mean embedding of the pixels under the
supplied point layer. The output is a single-band raster in ``[-1, 1]``, styled
with a blue-to-red ramp.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)

from ..eocore.descriptor import DEFAULT_NODATA
from ..eocore.ml import valid_mask_from_nodata
from ..eocore.similarity import cosine_similarity_map, reference_from_points
from ..eocore.streaming import DEFAULT_STREAM_TILE, map_windows, plan_windows
from ..eocore.tiling import Window
from . import _features, _raster
from ._common import (
    EoInferenceAlgorithm,
    add_auto_low_memory_params,
    as_advanced,
    read_descriptor,
    resolve_low_memory,
)
from ._styling import apply_continuous_ramp


class SimilarityAlgorithm(EoInferenceAlgorithm):
    """Cosine-similarity map to a set of reference points."""

    INPUT = "INPUT"
    POINTS = "POINTS"
    LOW_MEMORY = "LOW_MEMORY"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._dest_id: str = ""

    def name(self) -> str:
        return "similarity"

    def displayName(self) -> str:
        return "Similarity to reference"

    def shortHelpString(self) -> str:
        return (
            "Compute per-pixel cosine similarity of an embedding raster to the mean "
            "embedding of the pixels under a reference point layer. Output values run "
            "from -1 (opposite) to 1 (identical)."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster"))
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.POINTS,
                "Reference points",
                types=[QgsProcessing.SourceType.TypeVectorPoint],
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
        # Similarity streams byte-identically, so auto low-memory is safe by default.
        add_auto_low_memory_params(self, auto_default=True)
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Similarity"))

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
        source = self.parameterAsSource(parameters, self.POINTS, context)
        if source is None:
            raise QgsProcessingException("Could not read the reference point layer.")

        low_memory = resolve_low_memory(
            self,
            parameters,
            context,
            feedback,
            width=layer.width(),
            height=layer.height(),
            n_bands=layer.bandCount(),
        )
        if low_memory:
            result = self._similarity_streaming(parameters, context, feedback, path, source)
        else:
            result = self._similarity_whole(parameters, context, feedback, path, source)
        if result is None or feedback.isCanceled():
            return {}
        sim, geotransform, wkt = result

        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            sim,
            geotransform,
            wkt,
            nodata=DEFAULT_NODATA,
            band_names=["cosine_similarity"],
        )
        self._dest_id = output
        return {self.OUTPUT: output}

    def _resolve_rowcols(
        self,
        source: Any,
        grid: Any,
        wkt: str,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> list[tuple[int, int]]:
        dst_crs = QgsCoordinateReferenceSystem.fromWkt(wkt)
        rowcols = _features.sample_reference_rowcols(
            source, grid, dst_crs, context.transformContext()
        )
        if not rowcols:
            raise QgsProcessingException("No reference points fell inside the embedding raster.")
        feedback.pushInfo(f"Averaging {len(rowcols)} reference pixel(s).")
        return rowcols

    def _similarity_whole(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        path: str,
        source: Any,
    ) -> tuple[Any, Any, str] | None:
        """Default path: load the whole cube and compute the map in one shot."""
        raster = _raster.read_raster(path)
        rowcols = self._resolve_rowcols(source, raster.grid_spec(), raster.wkt, context, feedback)
        nodata = raster.nodata if raster.nodata is not None else DEFAULT_NODATA
        descriptor = read_descriptor(path, raster.cube.shape[0])
        try:
            reference = reference_from_points(
                raster.cube, rowcols, normalize=True, nodata=float(nodata)
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        sim = cosine_similarity_map(
            raster.cube, reference, assume_unit=descriptor.is_unit_normalized
        )
        mask = valid_mask_from_nodata(raster.cube, float(nodata))
        sim[~mask] = DEFAULT_NODATA
        return sim, raster.geotransform, raster.wkt

    def _similarity_streaming(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        path: str,
        source: Any,
    ) -> tuple[Any, Any, str] | None:
        """Low-memory path: stream tiles. Byte-identical to the whole-cube path.

        Cosine similarity is a per-pixel function of the fixed reference vector, so
        tiling the input changes nothing about the result — it only bounds memory to
        one tile of the wide embedding cube at a time.
        """
        info = _raster.read_raster_info(path)
        rowcols = self._resolve_rowcols(source, info.grid_spec(), info.wkt, context, feedback)
        nodata = float(info.nodata if info.nodata is not None else DEFAULT_NODATA)
        descriptor = read_descriptor(path, info.n_bands)
        windows = plan_windows(info.width, info.height, DEFAULT_STREAM_TILE)

        with _raster.open_window_reader(path) as read:
            vectors = [read(Window(col, row, 1, 1))[:, 0, 0] for row, col in rowcols]
            mini_cube = np.stack(vectors, axis=1)[:, :, np.newaxis]
            try:
                reference = reference_from_points(
                    mini_cube,
                    [(j, 0) for j in range(len(vectors))],
                    normalize=True,
                    nodata=nodata,
                )
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc

            def block_fn(tile: Any) -> Any:
                sim = cosine_similarity_map(
                    tile, reference, assume_unit=descriptor.is_unit_normalized
                ).astype(np.float32, copy=True)
                sim[~valid_mask_from_nodata(tile, nodata)] = DEFAULT_NODATA
                return sim[np.newaxis, :, :]

            sim_map = map_windows(
                windows,
                read,
                block_fn,
                out_bands=1,
                height=info.height,
                width=info.width,
                dtype=np.float32,
                fill=DEFAULT_NODATA,
                progress=feedback.setProgress,
                is_canceled=feedback.isCanceled,
            )
        if sim_map is None:
            return None
        return sim_map[0], info.geotransform, info.wkt

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        layer = QgsProcessingUtils.mapLayerFromString(self._dest_id, context)
        if layer is not None:
            apply_continuous_ramp(layer)
        return {self.OUTPUT: self._dest_id}
