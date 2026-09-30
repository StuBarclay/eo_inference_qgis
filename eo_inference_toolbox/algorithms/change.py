"""Change and trajectory metrics across two or more aligned embedding rasters.

Three modes: ``pairwise`` (distance first vs last), ``trajectory`` (summed
consecutive distances along the ordered series) and ``anomaly`` (distance of the
last raster from the mean of the earlier ones). All rasters must share the same
pixel grid.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qgis.core import (
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterRasterDestination,
    QgsProcessingUtils,
)

from ..eocore.change import (
    METRICS,
    anomaly_from_baseline,
    distance_map,
    mean_embedding,
    trajectory_magnitude,
)
from ..eocore.descriptor import DEFAULT_NODATA
from ..eocore.ml import valid_mask_from_nodata
from ..eocore.streaming import DEFAULT_STREAM_TILE, map_windows_multi, plan_windows
from . import _raster
from ._common import (
    EoInferenceAlgorithm,
    add_auto_low_memory_params,
    as_advanced,
    read_descriptor,
    resolve_low_memory,
)
from ._styling import apply_continuous_ramp

# Message shown when rasters differ and the user has not opted into resampling.
_MISALIGNED_MESSAGE = (
    "All embedding rasters must share the same pixel grid. Enable "
    "'Resample rasters onto the first raster's grid' to warp them."
)

MODES: tuple[str, ...] = ("pairwise", "trajectory", "anomaly")


class ChangeAlgorithm(EoInferenceAlgorithm):
    """Distance / trajectory / anomaly across embedding rasters."""

    INPUT = "INPUT"
    MODE = "MODE"
    METRIC = "METRIC"
    RESAMPLE = "RESAMPLE"
    LOW_MEMORY = "LOW_MEMORY"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()
        self._dest_id: str = ""

    def name(self) -> str:
        return "change"

    def displayName(self) -> str:
        return "Change / trajectory"

    def shortHelpString(self) -> str:
        return (
            "Measure change across two or more aligned embedding rasters (add them in "
            "time order):\n"
            "- pairwise: distance between the first and last raster;\n"
            "- trajectory: sum of consecutive distances along the series;\n"
            "- anomaly: distance of the last raster from the mean of the earlier ones."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT,
                "Embedding rasters (in time order)",
                layerType=QgsProcessing.SourceType.TypeRaster,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(self.MODE, "Mode", options=list(MODES), defaultValue=0)
        )
        self.addParameter(
            QgsProcessingParameterEnum(self.METRIC, "Metric", options=list(METRICS), defaultValue=0)
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.RESAMPLE,
                    "Resample rasters onto the first raster's grid if they differ",
                    defaultValue=False,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.LOW_MEMORY,
                    "Low-memory mode (stream the rasters in tiles instead of loading them whole)",
                    defaultValue=False,
                )
            )
        )
        # Change streams byte-identically, so auto low-memory is safe by default.
        add_auto_low_memory_params(self, auto_default=True)
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Change"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        layers = self.parameterAsLayerList(parameters, self.INPUT, context)
        if not layers or len(layers) < 2:
            raise QgsProcessingException("Add at least two embedding rasters.")
        mode = MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        metric = METRICS[self.parameterAsEnum(parameters, self.METRIC, context)]
        resample = self.parameterAsBool(parameters, self.RESAMPLE, context)
        sources = [layer.source() for layer in layers]

        first_layer = layers[0]
        low_memory = resolve_low_memory(
            self,
            parameters,
            context,
            feedback,
            width=first_layer.width(),
            height=first_layer.height(),
            n_bands=first_layer.bandCount(),
            n_cubes=len(layers),
        )
        feedback.pushInfo(
            f"Computing '{mode}' change with '{metric}' metric"
            f"{' (streamed)' if low_memory else ''}."
        )
        if low_memory:
            result = self._change_streaming(sources, mode, metric, resample, feedback)
        else:
            result = self._change_whole(sources, mode, metric, resample, feedback)
        if result is None or feedback.isCanceled():
            return {}
        change_map, geotransform, wkt = result

        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            change_map,
            geotransform,
            wkt,
            nodata=DEFAULT_NODATA,
            band_names=[f"{mode}_{metric}"],
        )
        self._dest_id = output
        return {self.OUTPUT: output}

    def _change_whole(
        self,
        sources: list[str],
        mode: str,
        metric: str,
        resample: bool,
        feedback: QgsProcessingFeedback,
    ) -> tuple[Any, Any, str] | None:
        """Default path: read every cube into RAM and compute the map in one shot."""
        rasters = [_raster.read_raster(src) for src in sources]
        rasters = self._align(rasters, sources, resample, feedback)
        first = rasters[0]
        cubes = [r.cube for r in rasters]
        descriptor = read_descriptor(sources[0], first.cube.shape[0])
        result = self._compute(mode, cubes, metric, descriptor.is_unit_normalized)
        nodata = first.nodata if first.nodata is not None else DEFAULT_NODATA
        mask = self._combined_mask(cubes, float(nodata))
        result[~mask] = DEFAULT_NODATA
        return result, first.geotransform, first.wkt

    def _change_streaming(
        self,
        sources: list[str],
        mode: str,
        metric: str,
        resample: bool,
        feedback: QgsProcessingFeedback,
    ) -> tuple[Any, Any, str] | None:
        """Low-memory path: stream one window from every raster at a time.

        Byte-identical to the whole-cube path: every change metric is a per-pixel
        function of the aligned inputs, so tiling changes nothing about the result —
        it only bounds peak memory to one tile per raster instead of holding all *N*
        wide embedding cubes at once (the pipeline's heaviest read). Rasters that need
        resampling are first warped to a temporary GeoTIFF on disk (GDAL streams the
        warp), so even the align step never materialises a full warped cube.
        """
        infos = [_raster.read_raster_info(src) for src in sources]
        first = infos[0]
        grid = first.grid_spec()
        nodata = float(first.nodata if first.nodata is not None else DEFAULT_NODATA)
        descriptor = read_descriptor(sources[0], first.n_bands)
        assume_unit = descriptor.is_unit_normalized
        windows = plan_windows(first.width, first.height, DEFAULT_STREAM_TILE)

        with contextlib.ExitStack() as stack:
            readers = [stack.enter_context(_raster.open_window_reader(sources[0]))]
            tmp_dir: str | None = None
            for index, (src, info) in enumerate(zip(sources[1:], infos[1:], strict=True), start=1):
                if _raster.infos_aligned(first, info):
                    readers.append(stack.enter_context(_raster.open_window_reader(src)))
                    continue
                if not resample:
                    raise QgsProcessingException(_MISALIGNED_MESSAGE)
                if tmp_dir is None:
                    tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                feedback.pushInfo(f"Resampling {src} onto the first raster's grid...")
                aligned_path = os.path.join(tmp_dir, f"aligned_{index}.tif")
                src_nodata = float(info.nodata if info.nodata is not None else DEFAULT_NODATA)
                _raster.warp_raster_to_grid_file(
                    src, grid, first.wkt, aligned_path, resample="nearest", src_nodata=src_nodata
                )
                readers.append(stack.enter_context(_raster.open_window_reader(aligned_path)))

            def block_fn(tiles: list[NDArray[Any]]) -> NDArray[Any]:
                result = self._compute(mode, tiles, metric, assume_unit).astype(
                    np.float32, copy=True
                )
                result[~self._combined_mask(tiles, nodata)] = DEFAULT_NODATA
                return result[np.newaxis, :, :]

            change_map = map_windows_multi(
                windows,
                readers,
                block_fn,
                out_bands=1,
                height=first.height,
                width=first.width,
                dtype=np.float32,
                fill=DEFAULT_NODATA,
                progress=feedback.setProgress,
                is_canceled=feedback.isCanceled,
            )
        if change_map is None:
            return None
        return change_map[0], first.geotransform, first.wkt

    def _align(
        self,
        rasters: list[Any],
        sources: list[str],
        resample: bool,
        feedback: QgsProcessingFeedback,
    ) -> list[Any]:
        """Ensure every raster shares the first raster's grid.

        By default a mismatch is a hard error (embeddings must be pixel-aligned to
        be comparable). If the user opts in, mismatched rasters are warped onto the
        first raster's grid with **nearest-neighbour** resampling — the only
        defensible choice for embeddings, since interpolating feature vectors (or
        blending the nodata sentinel into real pixels) would invent embeddings no
        model produced. The source's own nodata is carried through so those pixels
        stay nodata rather than being averaged in. Even so this is geometric
        alignment only, so the caller is trusting that resampled embeddings remain
        meaningful.
        """
        first = rasters[0]
        aligned = [first]
        for source, other in zip(sources[1:], rasters[1:], strict=True):
            if _raster.cubes_aligned(first, other):
                aligned.append(other)
                continue
            if not resample:
                raise QgsProcessingException(_MISALIGNED_MESSAGE)
            feedback.pushInfo(f"Resampling {source} onto the first raster's grid...")
            src_nodata = float(other.nodata if other.nodata is not None else DEFAULT_NODATA)
            aligned.append(
                _raster.warp_raster_to_grid(
                    source,
                    first.grid_spec(),
                    first.wkt,
                    resample="nearest",
                    src_nodata=src_nodata,
                )
            )
        return aligned

    def _compute(
        self,
        mode: str,
        cubes: list[NDArray[Any]],
        metric: str,
        assume_unit: bool,
    ) -> NDArray[Any]:
        if mode == "pairwise":
            return distance_map(cubes[0], cubes[-1], metric=metric, assume_unit=assume_unit)
        if mode == "trajectory":
            return trajectory_magnitude(cubes, metric=metric, assume_unit=assume_unit)
        baseline = mean_embedding(cubes[:-1])
        return anomaly_from_baseline(cubes[-1], baseline, metric=metric, assume_unit=assume_unit)

    def _combined_mask(self, cubes: list[NDArray[Any]], nodata: float) -> NDArray[Any]:
        mask = valid_mask_from_nodata(cubes[0], nodata)
        for cube in cubes[1:]:
            mask = mask & valid_mask_from_nodata(cube, nodata)
        return mask

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        layer = QgsProcessingUtils.mapLayerFromString(self._dest_id, context)
        if layer is not None:
            apply_continuous_ramp(layer)
        return {self.OUTPUT: self._dest_id}
