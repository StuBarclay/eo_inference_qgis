"""Prithvi-EO-2.0 backend: IBM + NASA multi-temporal ViT MAE, via TerraTorch.

Prithvi expects a fixed six-band input in the pre-training order (Blue, Green,
Red, narrow-NIR, SWIR1, SWIR2), nominally at 30 m from HLS. Pixels are
standardised per band as ``(DN - mean) / std`` using the statistics published in
the model's ``config.json`` (in raw HLS surface-reflectance DN units), exactly as
Prithvi's own preprocessing does — the pretrained backbone does not rescale or
normalise its input. It is multi-temporal — input shape ``(B, C, T, H, W)`` — but
the plugin feeds a single acquisition (``T = 1``). The backbone returns patch
tokens, which we reshape to a grid and upsample to the tile grid.

Torch and TerraTorch are imported lazily inside methods. The model build call is
isolated in ``_load_model`` and the forward pass in ``_encode_patches`` — the two
seams to confirm during the smoke test.

References
----------
- Weights + stats: Hugging Face ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M``
  (``config.json`` ``mean`` / ``std``), Apache-2.0.
- TerraTorch: https://github.com/IBM/terratorch
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..descriptor import EmbeddingDescriptor
from ..stac import PRITHVI_BANDS
from ._deps import optional_import_error
from ._pool import tokens_to_grid, upsample
from .base import (
    backend_info,
    descriptor_from_info,
    select_feature,
    unsupported_model_error,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._weights import ProgressFn
    from .base import TileMeta


# Prithvi-EO-2.0 standardises each band as ``(DN - mean) / std`` with the per-band
# statistics published in the model's ``config.json`` (see module docstring). The
# values below are for ibm-nasa-geospatial/Prithvi-EO-2.0-300M and are shared by
# the 600M / -TL variants — they are HLS dataset statistics, not model-specific.
# Keyed by the plugin's semantic band name in PRITHVI_BANDS order (the config's
# bands B02,B03,B04,B05,B06,B07 = Blue,Green,Red,narrow-NIR,SWIR1,SWIR2). A band
# absent from this map passes through unstandardised (mean 0, std 1) rather than
# raising, mirroring the Clay backend.
PRITHVI_BAND_STATS: dict[str, tuple[float, float]] = {
    "blue": (1087.0, 2248.0),
    "green": (1342.0, 2179.0),
    "red": (1433.0, 2178.0),
    "narrow_nir": (2734.0, 1850.0),
    "swir1": (1958.0, 1242.0),
    "swir2": (1363.0, 1049.0),
}

# Prithvi's ViT prepends a class token, so the backbone emits one token more than
# the patch grid. Allow a small margin (a variant might add a few register tokens)
# when reconciling the token count with the grid, but reject a gross mismatch —
# which almost always means the configured patch size disagrees with the
# checkpoint (or the encoder masked patches) and would otherwise silently scramble
# the grid. Mirrors the Clay guard.
_MAX_NON_PATCH_TOKENS = 8


class PrithviBackend:
    """Prithvi-EO-2.0 embedding backend (TerraTorch backbone)."""

    backend_id = "prithvi"
    display_name = "Prithvi-EO-2.0 (IBM + NASA)"

    def __init__(
        self,
        *,
        model_id: str = "prithvi_eo_v2_300",
        weights: str | Path | None = None,
        device: str | None = None,
        patch_size: int = 16,
        reflectance_scale: float = 1.0,
        upsampling: str = "nearest",
        cache_dir: str | Path | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self._model_id = model_id
        self._weights = weights
        self._device_pref = device
        self._patch_size = int(patch_size)
        # Prithvi standardises each band as ``(DN - mean) / std`` from the
        # published statistics, which are in raw HLS surface-reflectance DN. This
        # scale is an optional *pre*-standardisation multiplier into those DN units
        # and defaults to 1.0 (imagery already in DN — the normal case for fetched
        # HLS); set it only for imagery in other units (e.g. 1e4 for reflectance
        # already in [0, 1]).
        self._scale = float(reflectance_scale)
        self._upsampling = upsampling
        self._cache_dir = cache_dir
        self._progress = progress
        self._model: Any = None
        self._device: Any = None
        self._embedding_dim: int | None = None

    # -- interface ---------------------------------------------------------
    def descriptor(self) -> EmbeddingDescriptor:
        return descriptor_from_info(backend_info(self.backend_id), n_bands=self._embedding_dim)

    def required_bands(self) -> tuple[str, ...]:
        return PRITHVI_BANDS

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
        n_bands, height, width = arr.shape
        if n_bands != len(PRITHVI_BANDS):
            raise ValueError(
                f"Prithvi needs exactly {len(PRITHVI_BANDS)} bands {PRITHVI_BANDS}, got {n_bands}"
            )
        self._ensure_model()
        tokens = self._encode_patches(arr)
        grid_h = max(1, height // self._patch_size)
        grid_w = max(1, width // self._patch_size)
        self._check_token_count(int(tokens.shape[0]), grid_h, grid_w)
        grid = tokens_to_grid(tokens, grid_h, grid_w)
        self._embedding_dim = int(grid.shape[0])
        out: NDArray[Any] = upsample(grid, height, width, method=self._upsampling)
        return out

    def embed_tiles(self, images: Sequence[NDArray[Any]], meta: TileMeta) -> list[NDArray[Any]]:
        """Embed a batch of equal-shaped six-band tiles in one forward pass (E-2).

        Prithvi ignores ``meta``; equal-shaped tiles stack into one
        ``(B, C, T=1, H, W)`` input. Ragged batches fall back to per-tile.
        """
        arrs = [np.asarray(image, dtype=np.float32) for image in images]
        for arr in arrs:
            if arr.ndim != 3:
                raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
            if arr.shape[0] != len(PRITHVI_BANDS):
                raise ValueError(
                    f"Prithvi needs exactly {len(PRITHVI_BANDS)} bands {PRITHVI_BANDS}, "
                    f"got {arr.shape[0]}"
                )
        if len({arr.shape for arr in arrs}) != 1:
            return [self.embed_tile(arr, meta) for arr in arrs]
        self._ensure_model()
        batch = np.stack(arrs, axis=0)
        tokens = self._encode_patches_batch(batch)  # (B, n_tokens, D)
        _, height, width = arrs[0].shape
        grid_h = max(1, height // self._patch_size)
        grid_w = max(1, width // self._patch_size)
        self._check_token_count(int(tokens.shape[1]), grid_h, grid_w)
        outputs: list[NDArray[Any]] = []
        for sample in tokens:
            grid = tokens_to_grid(np.asarray(sample), grid_h, grid_w)
            self._embedding_dim = int(grid.shape[0])
            outputs.append(upsample(grid, height, width, method=self._upsampling))
        return outputs

    # -- model plumbing ----------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyTorch is required for the Prithvi backend; install the "
                "'inference' dependency tier"
            ) from exc
        self._device = torch.device(
            self._device_pref
            if self._device_pref
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = self._load_model()

    def _load_model(self) -> Any:
        """Build the Prithvi backbone via TerraTorch's registry.

        VERIFY during smoke test: registry name and ``build`` signature match the
        installed TerraTorch. An explicit local ``weights`` path is passed through
        to ``ckpt_path`` if provided.
        """
        from ._weights import resolve_local

        # Validate an explicit local checkpoint path up front (a clear
        # FileNotFoundError) before importing TerraTorch, mirroring Clay — an
        # unresolvable path would otherwise fail deep inside the registry build.
        local = resolve_local(self._weights)
        try:
            from terratorch.registry import (  # type: ignore[import-not-found]
                BACKBONE_REGISTRY,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                optional_import_error(
                    exc,
                    label="TerraTorch (the 'terratorch' package, needed for Prithvi)",
                    module="terratorch",
                    install_cmd="pip install terratorch",
                )
            ) from exc
        kwargs: dict[str, Any] = {"pretrained": local is None}
        if local is not None:
            kwargs["ckpt_path"] = str(local)
        # Signal the start of the (possibly long) first-run weight download that
        # TerraTorch performs internally, so a UI can show activity.
        if self._progress is not None and local is None:
            self._progress(0, None)
        # Fail loud with an actionable message if the registry name / build
        # signature has drifted in the installed TerraTorch (H-6) rather than
        # letting an opaque KeyError/AttributeError escape.
        try:
            model = BACKBONE_REGISTRY.build(self._model_id, **kwargs)
        except (AttributeError, KeyError, TypeError) as exc:
            raise unsupported_model_error(
                "prithvi",
                f"TerraTorch could not build backbone {self._model_id!r} ({exc})",
            ) from exc
        if not callable(model):
            raise unsupported_model_error(
                "prithvi", f"the built backbone {self._model_id!r} is not callable"
            )
        model.eval()
        model.to(self._device)
        # Inference is wrapped in ``torch.no_grad()`` at the call site; we avoid
        # ``torch.set_grad_enabled(False)`` here (it is a process-wide side effect).
        return model

    # -- inputs ------------------------------------------------------------
    def _band_stats(self, bands: Sequence[str]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return ``(means, stds)`` aligned to ``bands`` as ``(C, 1, 1)`` arrays.

        Values come from :data:`PRITHVI_BAND_STATS`; a band with no published
        statistic (or a zero std) falls back to identity (mean 0, std 1) — passed
        through unstandardised. The ``(C, 1, 1)`` shape broadcasts over both a
        single ``(C, H, W)`` tile and a ``(B, C, H, W)`` batch.
        """
        means: list[float] = []
        stds: list[float] = []
        for band in bands:
            stat = PRITHVI_BAND_STATS.get(band)
            if stat is None or stat[1] == 0.0:
                means.append(0.0)
                stds.append(1.0)
            else:
                means.append(stat[0])
                stds.append(stat[1])
        means_arr = np.asarray(means, dtype=np.float32).reshape(-1, 1, 1)
        stds_arr = np.asarray(stds, dtype=np.float32).reshape(-1, 1, 1)
        return means_arr, stds_arr

    def _standardize(self, arr: NDArray[Any]) -> NDArray[Any]:
        """Prithvi's expected input: per-band ``(DN - mean) / std`` (see module docstring).

        ``reflectance_scale`` is an optional pre-scale into DN units (default
        1.0). Accepts a ``(C, H, W)`` tile or a ``(B, C, H, W)`` batch; the
        ``(C, 1, 1)`` stats broadcast over either. Prithvi's input is the fixed
        six-band :data:`PRITHVI_BANDS` set, so the stats are keyed positionally to
        that order.
        """
        means, stds = self._band_stats(PRITHVI_BANDS)
        scaled = np.asarray(arr, dtype=np.float32) * self._scale
        standardized: NDArray[Any] = ((scaled - means) / stds).astype(np.float32, copy=False)
        return standardized

    def _check_token_count(self, n_tokens: int, grid_h: int, grid_w: int) -> None:
        """Guard the backbone's token count against the expected patch grid (H-6).

        ``tokens_to_grid`` drops leading class/register tokens, so a correct run
        has ``n_tokens`` equal to ``grid_h * grid_w`` plus a small number of
        leading tokens. A negative or large excess means the configured patch size
        disagrees with the checkpoint, the backbone masked patches, or the output
        is not a token sequence — without this guard the grid would be silently
        mis-sliced and scrambled.
        """
        expected = grid_h * grid_w
        extra = n_tokens - expected
        if extra < 0 or extra > _MAX_NON_PATCH_TOKENS:
            raise unsupported_model_error(
                "prithvi",
                f"the backbone returned {n_tokens} tokens but a {grid_h}x{grid_w} "
                f"patch grid expects {expected} (plus at most "
                f"{_MAX_NON_PATCH_TOKENS} leading class/register tokens); the "
                f"configured patch size ({self._patch_size}) likely disagrees with "
                "the checkpoint",
            )

    def _encode_patches(self, arr: NDArray[Any]) -> NDArray[Any]:
        """Forward pass; returns ``(n_tokens, D)`` tokens for the single time step."""
        import torch

        # (B=1, C, T=1, H, W)
        tensor = torch.from_numpy(self._standardize(arr)).unsqueeze(0).unsqueeze(2).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        feats = select_feature(output, "prithvi", index=-1)
        feats = feats.detach().to("cpu").float()
        if feats.ndim == 3:  # (B, n_tokens, D) -> drop batch
            feats = feats[0]
        if feats.ndim != 2:
            raise unsupported_model_error(
                "prithvi",
                f"expected the backbone to return a (batch, tokens, dim) or "
                f"(tokens, dim) tensor, got {feats.ndim} dimensions; pin a "
                "known-good terratorch version",
            )
        result: NDArray[Any] = feats.numpy()
        return result

    def _encode_patches_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        """Forward pass over a ``(B, C, H, W)`` batch -> ``(B, n_tokens, D)``.

        VERIFY during smoke test alongside :meth:`_encode_patches`: the backbone
        accepts a batch dimension > 1 (it does for the standard Prithvi builds).
        """
        import torch

        # (B, C, T=1, H, W)
        tensor = torch.from_numpy(self._standardize(batch)).unsqueeze(2).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        feats = select_feature(output, "prithvi", index=-1)
        feats = feats.detach().to("cpu").float()
        if feats.ndim == 2:  # (n_tokens, D) -> single-sample batch
            feats = feats.unsqueeze(0)
        if feats.ndim != 3:
            raise unsupported_model_error(
                "prithvi",
                f"expected the backbone to return a (batch, tokens, dim) tensor "
                f"for a batched forward pass, got {feats.ndim} dimensions; pin a "
                "known-good terratorch version",
            )
        result: NDArray[Any] = feats.numpy()
        return result
