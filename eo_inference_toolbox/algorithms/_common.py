"""Shared helpers and a base class for the EO Inference processing algorithms.

Everything QGIS-specific that more than one algorithm needs lives here: the
common base class (group id, translation, instance factory), the backend enum
plumbing, extent -> grid / bbox conversions, and reading the descriptor metadata
this plugin stamps into its embedding rasters.

This module imports ``qgis`` and (via :mod:`._raster`) the GDAL glue, but neither
pulls in a heavy library at import time: ``_raster`` defers ``osgeo`` to inside its
own functions, so importing this module stays cheap and cannot fail when GDAL is
absent.
"""

from __future__ import annotations

from typing import Any

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterNumber,
    QgsRectangle,
)
from qgis.PyQt.QtCore import QCoreApplication

from ..eocore.backends import BACKEND_IDS, BACKENDS, backend_info
from ..eocore.descriptor import DEFAULT_NODATA, EmbeddingDescriptor
from ..eocore.grid import GridSpec, grid_from_bounds
from ..eocore.streaming import (
    DEFAULT_MEMORY_BUDGET_MB,
    estimate_cube_bytes,
    should_auto_stream,
)
from ._raster import MD_BACKEND, MD_UNIT, read_metadata_item

# Fallback working CRS when the project has none (metres, global).
DEFAULT_CRS = "EPSG:3857"

# Extra metadata keys (imagery inputs + classifier outputs). The embedding
# descriptor keys (MD_BACKEND / MD_UNIT) live in :mod:`._raster`.
MD_BANDS = "EOINFER_BANDS"
MD_GSD = "EOINFER_GSD"
MD_COLLECTION = "EOINFER_COLLECTION"
MD_CLASSES = "EOINFER_CLASSES"

# Backend enum, in a stable display order for the GUI and the enum parameter.
BACKEND_ORDER: tuple[str, ...] = tuple(BACKEND_IDS)
BACKEND_LABELS: list[str] = [BACKENDS[bid].display_name for bid in BACKEND_ORDER]


def tr(text: str) -> str:
    """Translate a UI string in the ``eoinfer`` context."""
    return QCoreApplication.translate("eoinfer", text)


def as_advanced(param: Any) -> Any:
    """Flag a parameter as advanced (hidden behind the dialog's Advanced toggle)."""
    param.setFlags(param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
    return param


def add_auto_low_memory_params(algorithm: QgsProcessingAlgorithm, *, auto_default: bool) -> None:
    """Register the advanced auto-low-memory toggle + budget on an analytic.

    These sit alongside each analytic's own explicit "Low-memory mode" checkbox and
    let the plugin lift the RAM ceiling without the user ticking a box: when the
    estimated whole-cube footprint exceeds the budget, the streaming path is used.
    ``auto_default`` is on only where the streaming result is byte-identical to the
    whole-cube result (similarity / change / classify) and off where streaming
    changes an unsupervised *fit* and therefore the output (cluster / to_rgb) — a
    would-be OOM run there still gets a different-but-valid result, but only when
    the user opts in.
    """
    algorithm.addParameter(
        as_advanced(
            QgsProcessingParameterBoolean(
                "AUTO_LOW_MEMORY",
                "Auto low-memory (stream automatically when the input exceeds the budget below)",
                defaultValue=auto_default,
            )
        )
    )
    algorithm.addParameter(
        as_advanced(
            QgsProcessingParameterNumber(
                "MEMORY_BUDGET_MB",
                "Auto low-memory budget (MB of whole-cube memory to tolerate; 0 disables auto)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=DEFAULT_MEMORY_BUDGET_MB,
                minValue=0,
            )
        )
    )


def resolve_low_memory(
    algorithm: QgsProcessingAlgorithm,
    parameters: dict[str, Any],
    context: QgsProcessingContext,
    feedback: QgsProcessingFeedback,
    *,
    width: int,
    height: int,
    n_bands: int,
    n_cubes: int = 1,
) -> bool:
    """Whether an analytic should take its low-memory streaming path this run.

    ``True`` when the user ticks "Low-memory mode", or when "Auto low-memory" is on
    and the estimated whole-cube footprint (``n_cubes`` times the input cube, see
    :func:`eo_inference_toolbox.eocore.streaming.estimate_cube_bytes`) exceeds the
    per-run budget. The auto branch is logged through ``feedback`` so the switch is
    never silent. A zero budget disables the auto branch. This reads the shared
    parameter names registered by :func:`add_auto_low_memory_params` plus each
    analytic's own ``LOW_MEMORY`` checkbox.
    """
    if algorithm.parameterAsBool(parameters, "LOW_MEMORY", context):
        return True
    if not algorithm.parameterAsBool(parameters, "AUTO_LOW_MEMORY", context):
        return False
    budget_mb = algorithm.parameterAsInt(parameters, "MEMORY_BUDGET_MB", context)
    budget_bytes = int(budget_mb) * 1024 * 1024
    if not should_auto_stream(width, height, n_bands, budget_bytes=budget_bytes, n_cubes=n_cubes):
        return False
    est_mb = estimate_cube_bytes(width, height, n_bands, n_cubes=n_cubes) / (1024.0 * 1024.0)
    feedback.pushInfo(
        f"Auto low-memory: estimated input ~{est_mb:.0f} MB exceeds the {budget_mb} MB "
        "budget; streaming this raster in tiles instead of loading it whole."
    )
    return True


def backend_id_from_index(index: int) -> str:
    """Map an enum index back to a backend id."""
    if not 0 <= index < len(BACKEND_ORDER):
        raise ValueError(f"backend index {index} out of range")
    return BACKEND_ORDER[index]


def parse_band_list(text: str | None) -> tuple[str, ...] | None:
    """Parse a comma/space separated semantic-band string, or ``None`` if blank."""
    if not text:
        return None
    parts = [token.strip() for token in text.replace(",", " ").split()]
    bands = tuple(part for part in parts if part)
    return bands or None


def grid_from_extent(rect: QgsRectangle, pixel_size: float) -> GridSpec:
    """Build a north-up :class:`GridSpec` covering ``rect`` at ``pixel_size``."""
    bounds = (rect.xMinimum(), rect.yMinimum(), rect.xMaximum(), rect.yMaximum())
    return grid_from_bounds(bounds, float(pixel_size))


def wkt_for_crs(crs: QgsCoordinateReferenceSystem) -> str:
    """Return a WKT string for a CRS (never empty for a valid CRS)."""
    return str(crs.toWkt())


def read_descriptor(path: str, n_bands: int) -> EmbeddingDescriptor:
    """Reconstruct an :class:`EmbeddingDescriptor` from a raster's metadata.

    Falls back to a generic descriptor (backend id ``"unknown"``, not unit
    normalised) when the raster was not written by this plugin.
    """
    backend = read_metadata_item(path, MD_BACKEND) or "unknown"
    unit_raw = read_metadata_item(path, MD_UNIT)
    is_unit = str(unit_raw).strip() in {"1", "true", "True", "yes"}
    return EmbeddingDescriptor(
        backend_id=backend,
        n_bands=int(n_bands),
        is_unit_normalized=is_unit,
        nodata=DEFAULT_NODATA,
    )


def read_gsd(path: str, fallback: float) -> float:
    """Return the stamped ground sample distance, or ``fallback`` if absent."""
    raw = read_metadata_item(path, MD_GSD)
    if raw is None:
        return float(fallback)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(fallback)


class EoInferenceAlgorithm(QgsProcessingAlgorithm):
    """Base class: shared group, translation and instance factory."""

    def tr(self, text: str) -> str:
        return QCoreApplication.translate(type(self).__name__, text)

    def createInstance(self) -> EoInferenceAlgorithm:
        return type(self)()

    def group(self) -> str:
        return "EO Inference"

    def groupId(self) -> str:
        return "eoinference"

    # -- small shared helpers ------------------------------------------------
    def _backend_label_help(self) -> str:
        lines = [
            f"- {info.display_name}: {info.summary}"
            for info in (backend_info(bid) for bid in BACKEND_ORDER)
        ]
        return "Inference backend.\n" + "\n".join(lines)

    def _resolve_crs(self, crs: QgsCoordinateReferenceSystem) -> QgsCoordinateReferenceSystem:
        if crs is not None and crs.isValid():
            return crs
        return QgsCoordinateReferenceSystem(DEFAULT_CRS)

    def _stamp(self, descriptor: EmbeddingDescriptor) -> dict[str, str]:
        """Metadata dict stamping the descriptor into an output embedding raster."""
        return {
            MD_BACKEND: descriptor.backend_id,
            MD_UNIT: "1" if descriptor.is_unit_normalized else "0",
        }

    def _as_any(self, value: Any) -> Any:  # pragma: no cover - typing shim
        return value
