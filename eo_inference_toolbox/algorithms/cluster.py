"""Unsupervised K-means clustering of an embedding raster (scikit-learn).

K-means is fitted on a deterministic subsample of the valid pixels, then applied
to every valid pixel. The output is an integer cluster-id raster, styled with a
distinct colour per cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)

from ..eocore.descriptor import DEFAULT_NODATA
from ..eocore.ml import (
    cube_to_matrix,
    predict_in_chunks,
    scatter_to_grid,
    select_best_index,
    valid_mask_from_nodata,
)
from ..eocore.streaming import (
    DEFAULT_STREAM_TILE,
    collect_valid_sample,
    map_windows,
    partial_fit_stream,
    plan_windows,
)
from . import _raster, _sklearn
from ._common import (
    EoInferenceAlgorithm,
    add_auto_low_memory_params,
    as_advanced,
    resolve_low_memory,
)
from ._styling import apply_paletted

CLUSTER_NODATA = -1


class ClusterAlgorithm(EoInferenceAlgorithm):
    """K-means clustering over an embedding cube."""

    INPUT = "INPUT"
    N_CLUSTERS = "N_CLUSTERS"
    AUTO_K = "AUTO_K"
    K_MIN = "K_MIN"
    K_MAX = "K_MAX"
    SEED = "SEED"
    MAX_SAMPLES = "MAX_SAMPLES"
    LOW_MEMORY = "LOW_MEMORY"
    OUTPUT = "OUTPUT"

    # Silhouette is O(n^2); evaluate it on at most this many points per candidate.
    _SILHOUETTE_CAP = 5000

    def __init__(self) -> None:
        super().__init__()
        self._dest_id: str = ""
        self._cluster_values: list[int] = []

    def name(self) -> str:
        return "cluster"

    def displayName(self) -> str:
        return "Cluster embedding"

    def shortHelpString(self) -> str:
        return (
            "Group embedding pixels into K clusters with K-means (fitted on a "
            "subsample for speed). Handy for a quick unsupervised segmentation of a "
            "scene.\n\nRequires scikit-learn (install it from the toolbox dialog)."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster"))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_CLUSTERS,
                "Number of clusters (ignored when auto-select is on)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=8,
                minValue=2,
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.AUTO_K,
                    "Auto-select K by silhouette score",
                    defaultValue=False,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.K_MIN,
                    "Auto-select: smallest K to try",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=2,
                    minValue=2,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.K_MAX,
                    "Auto-select: largest K to try",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=12,
                    minValue=2,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.SEED,
                    "Random seed",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=0,
                    minValue=0,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.MAX_SAMPLES,
                    "Maximum pixels used to fit",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=100000,
                    minValue=1000,
                )
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterBoolean(
                    self.LOW_MEMORY,
                    "Low-memory mode (stream the raster in tiles; fits every pixel via "
                    "MiniBatch K-means)",
                    defaultValue=False,
                )
            )
        )
        # Cluster's streamed fit (MiniBatch K-means over every pixel) differs from the
        # whole-cube fit (K-means on a subsample), so it changes the output. Auto is
        # therefore off by default: a would-be-OOM run only streams when the user opts in.
        add_auto_low_memory_params(self, auto_default=False)
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Clusters"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        _sklearn.ensure_available()
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException("Could not load the input embedding raster.")
        path = layer.source()

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
            result = self._cluster_streaming(parameters, context, feedback, path)
        else:
            result = self._cluster_whole(parameters, context, feedback, path)
        if result is None or feedback.isCanceled():
            return {}
        clusters, geotransform, wkt = result

        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            clusters,
            geotransform,
            wkt,
            nodata=float(CLUSTER_NODATA),
            band_names=["cluster"],
        )
        self._dest_id = output
        return {self.OUTPUT: output}

    def _cluster_whole(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        path: str,
    ) -> tuple[Any, Any, str] | None:
        """Default path: read the whole cube, fit on a subsample, predict every pixel."""
        raster = _raster.read_raster(path)
        cube = raster.cube
        nodata = raster.nodata if raster.nodata is not None else DEFAULT_NODATA

        mask = valid_mask_from_nodata(cube, float(nodata))
        valid = cube_to_matrix(cube)[mask.reshape(-1)]
        if valid.shape[0] == 0:
            raise QgsProcessingException("The embedding raster has no valid pixels.")

        seed = self.parameterAsInt(parameters, self.SEED, context)
        max_samples = self.parameterAsInt(parameters, self.MAX_SAMPLES, context)

        sample = self._subsample(valid, max_samples, seed)
        kmeans = self._fit_kmeans(parameters, context, feedback, sample, valid.shape[0])
        if kmeans is None or feedback.isCanceled():
            return None

        preds = predict_in_chunks(valid, kmeans.predict)
        clusters = scatter_to_grid(preds, mask, fill=CLUSTER_NODATA, dtype=np.int32)
        self._cluster_values = sorted({int(v) for v in np.unique(preds)})
        return clusters, raster.geotransform, raster.wkt

    def _cluster_streaming(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        path: str,
    ) -> tuple[Any, Any, str] | None:
        """Low-memory path: stream tiles and fit ALL valid pixels via MiniBatch K-means.

        Only one tile of the wide embedding cube is in memory at a time. The fit
        (``MiniBatchKMeans.partial_fit`` over every tile) therefore sees the whole
        dataset — better than the default subsample — while the assembled output is
        a single int32 band.
        """
        info = _raster.read_raster_info(path)
        nodata = float(info.nodata if info.nodata is not None else DEFAULT_NODATA)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        max_samples = self.parameterAsInt(parameters, self.MAX_SAMPLES, context)
        windows = plan_windows(info.width, info.height, DEFAULT_STREAM_TILE)

        def valid_fn(tile: Any) -> Any:
            return valid_mask_from_nodata(tile, nodata)

        with _raster.open_window_reader(path) as read:
            k = self._resolve_k_streaming(
                parameters, context, feedback, read, windows, valid_fn, seed, max_samples
            )
            if k is None:
                return None
            batch = max(int(k), 4096)
            estimator = _sklearn.make_minibatch_kmeans(k, seed=seed, batch_size=batch)
            feedback.pushInfo(f"Fitting {k}-means over all valid pixels (MiniBatch, streamed).")
            n_valid, n_batches = partial_fit_stream(
                windows,
                read,
                valid_fn,
                lambda b: estimator.partial_fit(b),
                min_batch=batch,
                is_canceled=feedback.isCanceled,
            )
            if feedback.isCanceled():
                return None
            if n_batches == 0:
                raise QgsProcessingException("The embedding raster has no valid pixels.")
            feedback.pushInfo(f"Predicting clusters for {n_valid} pixel(s) (streamed).")
            present: set[int] = set()

            def block_fn(tile: Any) -> Any:
                flat = valid_fn(tile).reshape(-1)
                labels = np.full(flat.shape[0], CLUSTER_NODATA, dtype=np.int32)
                if flat.any():
                    preds = predict_in_chunks(cube_to_matrix(tile)[flat], estimator.predict)
                    preds = preds.astype(np.int32)
                    labels[flat] = preds
                    present.update(int(v) for v in np.unique(preds))
                return labels.reshape(1, tile.shape[1], tile.shape[2])

            clusters = map_windows(
                windows,
                read,
                block_fn,
                out_bands=1,
                height=info.height,
                width=info.width,
                dtype=np.int32,
                fill=CLUSTER_NODATA,
                progress=feedback.setProgress,
                is_canceled=feedback.isCanceled,
            )
        if clusters is None:
            return None
        self._cluster_values = sorted(present)
        return clusters, info.geotransform, info.wkt

    def _resolve_k_streaming(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        read: Any,
        windows: Any,
        valid_fn: Any,
        seed: int,
        max_samples: int,
    ) -> int | None:
        """K for the streaming path: the fixed value, or auto-select on a bounded sample."""
        if not self.parameterAsBool(parameters, self.AUTO_K, context):
            return self.parameterAsInt(parameters, self.N_CLUSTERS, context)
        sample, n_valid = collect_valid_sample(
            windows, read, valid_fn, max_samples=max_samples, seed=seed
        )
        if sample.shape[0] == 0:
            raise QgsProcessingException("The embedding raster has no valid pixels.")
        feedback.pushInfo(f"Auto-selecting K on {sample.shape[0]} of {n_valid} sampled pixel(s).")
        model = self._auto_select_k(parameters, context, feedback, sample, seed)
        if model is None:
            return None
        return int(model.n_clusters)

    def _fit_kmeans(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        sample: Any,
        n_valid: int,
    ) -> Any:
        """Fit K-means, auto-selecting K by silhouette when the user opts in."""
        seed = self.parameterAsInt(parameters, self.SEED, context)
        auto_k = self.parameterAsBool(parameters, self.AUTO_K, context)
        if not auto_k:
            n_clusters = self.parameterAsInt(parameters, self.N_CLUSTERS, context)
            feedback.pushInfo(
                f"Fitting {n_clusters}-means on {sample.shape[0]} of {n_valid} pixel(s)."
            )
            kmeans = _sklearn.make_kmeans(n_clusters, seed=seed)
            kmeans.fit(sample)
            return kmeans
        return self._auto_select_k(parameters, context, feedback, sample, seed)

    def _auto_select_k(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        sample: Any,
        seed: int,
    ) -> Any:
        k_min = self.parameterAsInt(parameters, self.K_MIN, context)
        k_max = self.parameterAsInt(parameters, self.K_MAX, context)
        if k_max < k_min:
            raise QgsProcessingException("Auto-select: largest K must be >= smallest K.")
        # Silhouette needs at least 2 clusters and more samples than clusters.
        k_max = min(k_max, sample.shape[0] - 1)
        if k_max < k_min:
            raise QgsProcessingException("Too few sample pixels for the requested K range.")
        sil_sample = self._subsample(sample, self._SILHOUETTE_CAP, seed)
        candidates = list(range(k_min, k_max + 1))
        fitted: list[Any] = []
        scores: list[float] = []
        for k in candidates:
            if feedback.isCanceled():
                return None
            model = _sklearn.make_kmeans(k, seed=seed)
            model.fit(sample)
            score = _sklearn.silhouette(sil_sample, model.predict(sil_sample))
            feedback.pushInfo(f"K={k}: silhouette={score:.4f}")
            fitted.append(model)
            scores.append(score)
        best = select_best_index(scores, mode="max")
        feedback.pushInfo(f"Auto-selected K={candidates[best]} (silhouette={scores[best]:.4f}).")
        return fitted[best]

    def _subsample(self, valid: Any, max_samples: int, seed: int) -> Any:
        if valid.shape[0] <= max_samples:
            return valid
        rng = np.random.default_rng(seed)
        index = rng.choice(valid.shape[0], size=max_samples, replace=False)
        return valid[index]

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        layer = QgsProcessingUtils.mapLayerFromString(self._dest_id, context)
        if layer is not None and self._cluster_values:
            apply_paletted(layer, self._cluster_values)
        return {self.OUTPUT: self._dest_id}
