"""Describes an embedding source, independent of how the vectors were produced.

The descriptor is the contract between a backend (which knows *how many* bands it
emits and whether they are already unit-length) and the analytics (which must
L2-normalise non-unit vectors before a cosine metric is valid). Stamping the
backend id into a saved model also stops a model trained on, say, Clay vectors
from being silently reused on Prithvi vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sentinel written into embedding rasters for pixels with no valid input.
DEFAULT_NODATA = -9999.0


@dataclass(frozen=True)
class BackendCapabilities:
    """Coarse capability flags used by the GUI and dependency checks."""

    needs_gpu: bool = False
    needs_auth: bool = False
    extra_deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmbeddingDescriptor:
    """Metadata describing one embedding source (a backend or a loaded raster)."""

    backend_id: str
    n_bands: int
    is_unit_normalized: bool = False
    native_dtype: str = "float32"
    nodata: float = DEFAULT_NODATA
    resolution_m: float | None = None
    temporal: bool = False
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def __post_init__(self) -> None:
        if not self.backend_id:
            raise ValueError("backend_id must be a non-empty string")
        if self.n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {self.n_bands}")
