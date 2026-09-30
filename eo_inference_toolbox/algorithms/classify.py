"""Supervised classification of an embedding raster with scikit-learn.

Either train a fresh classifier from a labelled point/polygon layer (optionally
saving the model), or load a previously trained model and apply it. Saved models
carry the backend id and band count, so a model trained on one backend's vectors
cannot be silently applied to another's. An optional confidence raster records
the winning class probability.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)

from ..eocore.descriptor import DEFAULT_NODATA
from ..eocore.ml import (
    classification_metrics,
    confusion_matrix,
    cube_to_matrix,
    encode_labels,
    format_accuracy_report,
    predict_in_chunks,
    scatter_to_grid,
    train_val_split,
    valid_mask_from_nodata,
)
from ..eocore.model_io import check_compatible, load_model, model_envelope, save_model
from ..eocore.streaming import DEFAULT_STREAM_TILE, map_windows, plan_windows
from . import _features, _raster, _sklearn
from ._common import (
    MD_CLASSES,
    EoInferenceAlgorithm,
    add_auto_low_memory_params,
    as_advanced,
    read_descriptor,
    resolve_low_memory,
)
from ._styling import apply_paletted

CLASS_NODATA = -1


class ClassifyAlgorithm(EoInferenceAlgorithm):
    """Train / apply a supervised classifier over an embedding cube."""

    INPUT = "INPUT"
    TRAINING = "TRAINING"
    CLASS_FIELD = "CLASS_FIELD"
    CLASSIFIER = "CLASSIFIER"
    N_ESTIMATORS = "N_ESTIMATORS"
    SEED = "SEED"
    LOAD_MODEL = "LOAD_MODEL"
    OUTPUT = "OUTPUT"
    OUTPUT_MODEL = "OUTPUT_MODEL"
    CONFIDENCE = "CONFIDENCE"
    ACCURACY_REPORT = "ACCURACY_REPORT"
    VAL_FRACTION = "VAL_FRACTION"
    LOW_MEMORY = "LOW_MEMORY"

    def __init__(self) -> None:
        super().__init__()
        self._dest_id: str = ""
        self._class_values: list[int] = []

    def name(self) -> str:
        return "classify"

    def displayName(self) -> str:
        return "Classify embedding"

    def shortHelpString(self) -> str:
        return (
            "Classify an embedding raster. Provide a labelled point/polygon layer to "
            "train a Random Forest or Gradient Boosting model (optionally saved for "
            "reuse), or load a saved model to apply it. Optionally write a confidence "
            "raster.\n\nRequires scikit-learn (install it from the toolbox dialog)."
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, "Embedding raster"))
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TRAINING,
                "Training samples (points or polygons)",
                types=[
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorPolygon,
                ],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CLASS_FIELD,
                "Class field",
                parentLayerParameterName=self.TRAINING,
                type=QgsProcessingParameterField.DataType.Any,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.CLASSIFIER,
                "Classifier",
                options=list(_sklearn.CLASSIFIERS),
                defaultValue=0,
            )
        )
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.N_ESTIMATORS,
                    "Number of trees (random forest)",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=200,
                    minValue=1,
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
            QgsProcessingParameterFile(
                self.LOAD_MODEL,
                "Load a saved model (blank = train a new one)",
                optional=True,
            )
        )
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, "Classification"))
        model_out = QgsProcessingParameterFileDestination(
            self.OUTPUT_MODEL,
            "Save trained model",
            fileFilter="Trained model (*.pkl)",
            optional=True,
            createByDefault=False,
        )
        self.addParameter(as_advanced(model_out))
        conf_out = QgsProcessingParameterRasterDestination(
            self.CONFIDENCE,
            "Confidence (winning probability)",
            optional=True,
            createByDefault=False,
        )
        self.addParameter(as_advanced(conf_out))
        report_out = QgsProcessingParameterFileDestination(
            self.ACCURACY_REPORT,
            "Held-out accuracy report (blank = skip)",
            fileFilter="Text report (*.txt)",
            optional=True,
            createByDefault=False,
        )
        self.addParameter(as_advanced(report_out))
        self.addParameter(
            as_advanced(
                QgsProcessingParameterNumber(
                    self.VAL_FRACTION,
                    "Validation fraction for the accuracy report",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=0.2,
                    minValue=0.05,
                    maxValue=0.5,
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
        # Classify streams byte-identically (training collection matches pixel-for-pixel
        # and the fitted estimator is applied per pixel), so auto low-memory is safe by default.
        add_auto_low_memory_params(self, auto_default=True)

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
        info = _raster.read_raster_info(path)
        descriptor = read_descriptor(path, info.n_bands)
        want_confidence = bool(self.parameterAsOutputLayer(parameters, self.CONFIDENCE, context))

        low_memory = resolve_low_memory(
            self,
            parameters,
            context,
            feedback,
            width=info.width,
            height=info.height,
            n_bands=info.n_bands,
        )
        if low_memory:
            result = self._classify_streaming(
                parameters, context, feedback, path, info, descriptor, want_confidence
            )
        else:
            result = self._classify_whole(
                parameters, context, feedback, path, info, descriptor, want_confidence
            )
        if result is None or feedback.isCanceled():
            return {}
        classification, confidence, classes = result

        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        _raster.write_geotiff(
            output,
            classification,
            info.geotransform,
            info.wkt,
            nodata=float(CLASS_NODATA),
            band_names=["class"],
            metadata={MD_CLASSES: json.dumps([str(c) for c in classes])},
        )
        self._dest_id = output
        results: dict[str, Any] = {self.OUTPUT: output}
        if confidence is not None:
            conf_out = self.parameterAsOutputLayer(parameters, self.CONFIDENCE, context)
            _raster.write_geotiff(
                conf_out,
                confidence,
                info.geotransform,
                info.wkt,
                nodata=DEFAULT_NODATA,
                band_names=["confidence"],
            )
            results[self.CONFIDENCE] = conf_out
        return results

    def _classify_whole(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        path: str,
        info: Any,
        descriptor: Any,
        want_confidence: bool,
    ) -> tuple[Any, Any, list[Any]] | None:
        """Default path: load the whole cube, predict every valid pixel in one shot."""
        raster = _raster.read_raster(path)
        cube = raster.cube
        nodata = float(info.nodata if info.nodata is not None else DEFAULT_NODATA)
        mask = valid_mask_from_nodata(cube, nodata)
        valid = cube_to_matrix(cube)[mask.reshape(-1)]
        if valid.shape[0] == 0:
            raise QgsProcessingException("The embedding raster has no valid pixels.")

        loaded = self._maybe_load_model(parameters, context, feedback, descriptor, info.n_bands)
        if loaded is not None:
            estimator, classes = loaded
        else:
            source, class_field = self._training_source(parameters, context)
            dst_crs = QgsCoordinateReferenceSystem.fromWkt(raster.wkt)
            features, labels = _features.collect_training_samples(
                cube,
                raster.grid_spec(),
                source,
                class_field,
                dst_crs,
                context.transformContext(),
                nodata=nodata,
            )
            estimator, classes = self._fit_from_samples(
                parameters, context, feedback, features, labels, descriptor, info.n_bands
            )
        if feedback.isCanceled():
            return None

        feedback.pushInfo(f"Classifying {valid.shape[0]} pixel(s)...")
        preds = predict_in_chunks(valid, estimator.predict)
        classification = scatter_to_grid(preds, mask, fill=CLASS_NODATA, dtype=np.int32)
        self._class_values = sorted({int(v) for v in np.unique(preds)})

        confidence = None
        if want_confidence:
            if hasattr(estimator, "predict_proba"):
                proba = predict_in_chunks(valid, estimator.predict_proba)
                confidence = scatter_to_grid(
                    proba.max(axis=1), mask, fill=DEFAULT_NODATA, dtype=np.float32
                )
            else:
                feedback.pushWarning("Classifier has no probabilities; skipping confidence.")
        return classification, confidence, classes

    def _classify_streaming(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        path: str,
        info: Any,
        descriptor: Any,
        want_confidence: bool,
    ) -> tuple[Any, Any, list[Any]] | None:
        """Low-memory path: stream tiles for both training and the scene prediction.

        The class map is byte-identical to the whole-cube path (prediction is a
        per-pixel function of the fixed estimator). Training reads only each labelled
        feature's bounding-box window, and the whole-scene predict/confidence passes
        hold one tile of the wide cube at a time instead of the entire cube.
        """
        nodata = float(info.nodata if info.nodata is not None else DEFAULT_NODATA)
        windows = plan_windows(info.width, info.height, DEFAULT_STREAM_TILE)

        with _raster.open_window_reader(path) as read:
            loaded = self._maybe_load_model(parameters, context, feedback, descriptor, info.n_bands)
            if loaded is not None:
                estimator, classes = loaded
            else:
                source, class_field = self._training_source(parameters, context)
                dst_crs = QgsCoordinateReferenceSystem.fromWkt(info.wkt)
                features, labels = _features.collect_training_samples_windowed(
                    read,
                    info.grid_spec(),
                    source,
                    class_field,
                    dst_crs,
                    context.transformContext(),
                    n_bands=info.n_bands,
                    nodata=nodata,
                )
                estimator, classes = self._fit_from_samples(
                    parameters, context, feedback, features, labels, descriptor, info.n_bands
                )
            if feedback.isCanceled():
                return None

            present: set[int] = set()

            def class_block(tile: Any) -> Any:
                flat = valid_mask_from_nodata(tile, nodata).reshape(-1)
                out = np.full(flat.shape[0], CLASS_NODATA, dtype=np.int32)
                if flat.any():
                    preds = predict_in_chunks(cube_to_matrix(tile)[flat], estimator.predict)
                    preds = preds.astype(np.int32)
                    out[flat] = preds
                    present.update(int(v) for v in np.unique(preds))
                return out.reshape(1, tile.shape[1], tile.shape[2])

            feedback.pushInfo("Classifying scene (streamed)...")
            classification = map_windows(
                windows,
                read,
                class_block,
                out_bands=1,
                height=info.height,
                width=info.width,
                dtype=np.int32,
                fill=CLASS_NODATA,
                progress=feedback.setProgress,
                is_canceled=feedback.isCanceled,
            )
            if classification is None:
                return None

            confidence = None
            if want_confidence:
                confidence = self._confidence_streaming(
                    read, windows, info, estimator, nodata, feedback
                )
                if feedback.isCanceled():
                    return None
        self._class_values = sorted(present)
        conf_grid = confidence[0] if confidence is not None else None
        return classification[0], conf_grid, classes

    def _confidence_streaming(
        self,
        read: Any,
        windows: Any,
        info: Any,
        estimator: Any,
        nodata: float,
        feedback: QgsProcessingFeedback,
    ) -> Any:
        """Stream a winning-probability raster, or ``None`` if the model has none."""
        if not hasattr(estimator, "predict_proba"):
            feedback.pushWarning("Classifier has no probabilities; skipping confidence.")
            return None

        def conf_block(tile: Any) -> Any:
            flat = valid_mask_from_nodata(tile, nodata).reshape(-1)
            out = np.full(flat.shape[0], DEFAULT_NODATA, dtype=np.float32)
            if flat.any():
                proba = predict_in_chunks(cube_to_matrix(tile)[flat], estimator.predict_proba)
                out[flat] = proba.max(axis=1).astype(np.float32)
            return out.reshape(1, tile.shape[1], tile.shape[2])

        feedback.pushInfo("Computing confidence (streamed)...")
        return map_windows(
            windows,
            read,
            conf_block,
            out_bands=1,
            height=info.height,
            width=info.width,
            dtype=np.float32,
            fill=DEFAULT_NODATA,
            progress=feedback.setProgress,
            is_canceled=feedback.isCanceled,
        )

    # -- estimator acquisition ----------------------------------------------
    def _maybe_load_model(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        descriptor: Any,
        n_bands: int,
    ) -> tuple[Any, list[Any]] | None:
        """Load + compatibility-check a saved model, or ``None`` to train a fresh one."""
        model_path = self.parameterAsFile(parameters, self.LOAD_MODEL, context)
        if not model_path:
            return None
        # load_model / check_compatible fail closed with ValueError (unreadable /
        # wrong-format / mismatched model) and load can hit OSError on a bad path;
        # surface either as a clean processing error instead of a raw traceback.
        try:
            envelope = load_model(model_path)
            check_compatible(envelope, backend_id=descriptor.backend_id, n_bands=n_bands)
        except (ValueError, OSError) as exc:
            raise QgsProcessingException(
                f"Could not load the saved model at '{model_path}': {exc}"
            ) from exc
        feedback.pushInfo(f"Loaded model trained on backend '{envelope.get('backend_id')}'.")
        return envelope["estimator"], list(envelope.get("classes") or [])

    def _training_source(
        self, parameters: dict[str, Any], context: QgsProcessingContext
    ) -> tuple[Any, str]:
        source = self.parameterAsSource(parameters, self.TRAINING, context)
        class_field = self.parameterAsString(parameters, self.CLASS_FIELD, context)
        if source is None or not class_field:
            raise QgsProcessingException(
                "Provide training samples and a class field, or load a saved model."
            )
        return source, class_field

    def _fit_from_samples(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        features: Any,
        labels: list[Any],
        descriptor: Any,
        n_bands: int,
    ) -> tuple[Any, list[Any]]:
        """Fit (and optionally save / score) a classifier from collected samples."""
        if features.shape[0] == 0:
            raise QgsProcessingException("No training pixels fell on valid embedding cells.")
        codes, classes = encode_labels(labels)
        algorithm = _sklearn.CLASSIFIERS[self.parameterAsEnum(parameters, self.CLASSIFIER, context)]
        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        # Opt-in held-out accuracy report, computed BEFORE the final fit so the
        # estimate uses data the reported model never saw. The shipped estimator
        # is still fit on all samples below (best use of scarce training data).
        self._maybe_accuracy_report(
            parameters, context, feedback, features, codes, classes, algorithm, n_estimators, seed
        )
        estimator = _sklearn.make_classifier(algorithm, n_estimators=n_estimators, seed=seed)
        feedback.pushInfo(
            f"Training {algorithm} on {features.shape[0]} samples, {len(classes)} class(es)."
        )
        estimator.fit(features, codes)
        self._save_model(parameters, context, feedback, estimator, descriptor, n_bands, classes)
        return estimator, list(classes)

    def _maybe_accuracy_report(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        features: Any,
        codes: Any,
        classes: list[Any],
        algorithm: str,
        n_estimators: int,
        seed: int,
    ) -> None:
        report_out = self.parameterAsFileOutput(parameters, self.ACCURACY_REPORT, context)
        if not report_out:
            return
        n_samples = int(features.shape[0])
        if n_samples < 4:
            feedback.pushWarning("Too few training samples for a held-out report; skipping.")
            return
        val_fraction = self.parameterAsDouble(parameters, self.VAL_FRACTION, context)
        train_idx, val_idx = train_val_split(n_samples, val_fraction, seed=seed)
        scorer = _sklearn.make_classifier(algorithm, n_estimators=n_estimators, seed=seed)
        scorer.fit(features[train_idx], codes[train_idx])
        preds = predict_in_chunks(features[val_idx], scorer.predict)
        cm = confusion_matrix(codes[val_idx], preds, len(classes))
        metrics = classification_metrics(cm)
        text = format_accuracy_report(metrics, cm, classes, len(train_idx), len(val_idx))
        with open(report_out, "w", encoding="utf-8") as handle:
            handle.write(text)
        feedback.pushInfo(
            f"Held-out overall accuracy: {metrics['overall_accuracy']:.3f} "
            f"({len(val_idx)} validation samples). Report: {report_out}"
        )

    def _save_model(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        estimator: Any,
        descriptor: Any,
        n_bands: int,
        classes: list[Any],
    ) -> None:
        model_out = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
        if not model_out:
            return
        envelope = model_envelope(
            estimator,
            backend_id=descriptor.backend_id,
            n_bands=n_bands,
            kind="classifier",
            classes=classes,
        )
        save_model(model_out, envelope)
        feedback.pushInfo(f"Saved model to {model_out}.")

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        layer = QgsProcessingUtils.mapLayerFromString(self._dest_id, context)
        if layer is not None and self._class_values:
            apply_paletted(layer, self._class_values)
        return {self.OUTPUT: self._dest_id}
