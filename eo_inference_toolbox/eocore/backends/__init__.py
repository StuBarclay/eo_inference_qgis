"""Inference backends: Clay, Prithvi-EO-2.0 and SatlasPretrain.

Only :mod:`.base` is import-safe everywhere — it holds the protocol, the tile
metadata dataclass and a torch-free registry of backend *facts* (dimensions,
licences, pip packages) that the GUI and docs read without importing torch.

The concrete backends (:mod:`.clay`, :mod:`.prithvi`, :mod:`.satlas`) import
torch and the model packages lazily, inside methods, so importing this package
never pulls in a heavy dependency.
"""

from __future__ import annotations

from .base import (
    BACKEND_IDS,
    BACKENDS,
    BackendInfo,
    BackendLoader,
    BatchEmbeddingBackend,
    EmbeddingBackend,
    TileMeta,
    backend_info,
    band_requirement_error,
    embed_tile_batch,
    load_backend,
    register_backend,
    registered_backends,
    scale_sanity_warning,
)

__all__ = [
    "BACKENDS",
    "BACKEND_IDS",
    "BackendInfo",
    "BackendLoader",
    "BatchEmbeddingBackend",
    "EmbeddingBackend",
    "TileMeta",
    "backend_info",
    "band_requirement_error",
    "embed_tile_batch",
    "load_backend",
    "register_backend",
    "registered_backends",
    "scale_sanity_warning",
]
