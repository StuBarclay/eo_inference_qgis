"""Clay v1.5 backend: a multi-sensor ViT masked-autoencoder encoder.

Clay consumes a "datacube" — pixels plus metadata (acquisition time, tile
centre lat/lon, ground sample distance, and the central wavelength of each
band) — and returns patch tokens, which we reshape to a grid and upsample to the
tile's pixel resolution. Pixels are standardised per band as ``(DN - mean) / std``
using the per-sensor statistics vendored in ``clay_assets/metadata.yaml``, exactly
as Clay's own embedding path expects (see :meth:`ClayBackend._standardize`).

Everything torch-specific is imported lazily inside methods, so importing this
module never requires torch. The single model forward pass is isolated in
``_encode_patches`` and guarded with an actionable error, because the exact Clay
package API should be confirmed against the installed version during the smoke
test (see docs/SMOKE_TEST.md).

References
----------
- Clay model + docs: https://github.com/Clay-foundation/model
- Weights: Hugging Face ``made-with-clay/Clay`` (``v1.5/clay-v1.5.ckpt``), Apache-2.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..descriptor import EmbeddingDescriptor
from ._deps import optional_import_error
from ._pool import tokens_to_grid, upsample
from .base import (
    TileMeta,
    backend_info,
    descriptor_from_info,
    require_model_attr,
    select_feature,
    unsupported_model_error,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._weights import ProgressFn

#: Hugging Face location of the default Clay checkpoint (shared with the wizard's
#: pre-download action so there is a single source of truth).
CLAY_REPO = "made-with-clay/Clay"
# The checkpoint lives in the repo's ``v1.5/`` subfolder, not at the root, so the
# path passed to huggingface_hub MUST include it (a bare "clay-v1.5.ckpt" 404s).
# The file is 5_158_764_629 bytes (~5.16 GB). Its git-LFS sha256 is
# 21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0 — confirmed
# 2026-09-29 against Hugging Face's authoritative LFS pointer for this path.
# It is nonetheless left out of BackendInfo.weights_sha256 (kept None) on purpose:
# hf_download re-hashes the file on *every* resolve, including cache hits, so
# pinning it would re-read ~5 GB on each backend load for no added safety — HF's
# LFS layer already content-addresses the download, and a user-supplied local
# checkpoint bypasses verification entirely. The value is published in the user
# docs (SMOKE_TEST.md) for anyone who wants to verify a download by hand.
CLAY_CHECKPOINT = "v1.5/clay-v1.5.ckpt"

# Clay's ``configs/metadata.yaml`` vendored beside this module. The checkpoint
# saves a *relative* ``metadata_path`` (default "configs/metadata.yaml") that
# only resolves when the working directory is the Clay repo root; inside QGIS it
# is not, so ``load_from_checkpoint`` crashes with FileNotFoundError. We pass the
# absolute path to this vendored copy instead. See NOTICE for attribution.
CLAY_METADATA_ASSET = ("clay_assets", "metadata.yaml")

# Fallback central wavelength (micrometres) per semantic band, for Sentinel-2.
# The live values are read from the ``wavelength`` block of the vendored metadata
# (see :meth:`ClayBackend._band_waves`); this table is only a backstop for a band
# with no published wavelength, and a single source of truth for the values.
WAVELENGTHS_UM: dict[str, float] = {
    "blue": 0.493,
    "green": 0.560,
    "red": 0.665,
    "nir": 0.842,
    "narrow_nir": 0.865,
    "swir1": 1.610,
    "swir2": 2.190,
}

# Clay prepends a single class token to the patch sequence, so the encoder emits
# one token more than the patch grid. Allow a small margin (a future variant might
# add a few register tokens) when reconciling the token count with the grid, but
# reject a gross mismatch — which almost always means the configured patch size
# disagrees with the checkpoint and would otherwise silently scramble the grid.
_MAX_NON_PATCH_TOKENS = 8

# Clay's normalisation statistics are keyed by sensor/platform. The plugin's Clay
# default collection is Sentinel-2 L2A (and WAVELENGTHS_UM above is Sentinel-2), so
# per-band mean/std are read from this block of the vendored metadata.
CLAY_METADATA_PLATFORM = "sentinel-2-l2a"

# Plugin semantic band name -> the band key used in the "sentinel-2-l2a" block of
# Clay's metadata. Three names differ (narrow-NIR and the two SWIRs); the rest
# match. A band absent from this map is looked up directly by its own name, then
# falls back to identity stats (mean 0, std 1) — i.e. passed through
# unstandardised — if it is still unknown.
_S2_METADATA_KEY: dict[str, str] = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir",
    "narrow_nir": "nir08",
    "swir1": "swir16",
    "swir2": "swir22",
}


class ClayBackend:
    """Clay v1.5 embedding backend."""

    backend_id = "clay"
    display_name = "Clay v1.5"

    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        device: str | None = None,
        patch_size: int = 8,
        reflectance_scale: float = 1.0,
        upsampling: str = "nearest",
        cache_dir: str | Path | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self._weights = weights
        self._device_pref = device
        self._patch_size = int(patch_size)
        # Clay standardises each band as ``(DN - mean) / std`` from the vendored
        # metadata, whose statistics are in raw surface-reflectance DN. This scale
        # is an optional *pre*-standardisation multiplier into those DN units and
        # defaults to 1.0 (imagery already in DN — the normal case for the fetched
        # Sentinel-2 L2A); set it only for imagery in other units (e.g. 1e4 for
        # reflectance already in [0, 1]).
        self._scale = float(reflectance_scale)
        self._upsampling = upsampling
        self._cache_dir = cache_dir
        self._progress = progress
        self._model: Any = None
        self._device: Any = None
        self._embedding_dim: int | None = None
        # Per-band mean/std for the default sensor, parsed from the vendored
        # metadata on first use and cached (see :meth:`_sensor_stats`).
        self._stats_cache: dict[str, dict[str, float]] | None = None

    # -- interface ---------------------------------------------------------
    def descriptor(self) -> EmbeddingDescriptor:
        info = backend_info(self.backend_id)
        return descriptor_from_info(info, n_bands=self._embedding_dim)

    def required_bands(self) -> tuple[str, ...]:
        return backend_info(self.backend_id).default_bands

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        """Embed a ``(C_in, h, w)`` tile to a ``(D, h, w)`` cube."""
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
        _, height, width = arr.shape
        self._ensure_model()
        tokens = self._encode_patches(arr, meta)  # (n_tokens, D)
        grid_h = max(1, height // self._patch_size)
        grid_w = max(1, width // self._patch_size)
        self._check_token_count(int(tokens.shape[0]), grid_h, grid_w)
        grid = tokens_to_grid(tokens, grid_h, grid_w)
        self._embedding_dim = int(grid.shape[0])
        out: NDArray[Any] = upsample(grid, height, width, method=self._upsampling)
        return out

    def embed_tiles(self, images: Sequence[NDArray[Any]], meta: TileMeta) -> list[NDArray[Any]]:
        """Embed a batch of equal-shaped tiles in one encoder pass (E-2).

        All tiles share ``meta`` (the embed algorithm uses a scene-centre meta),
        so they stack into one ``(B, C, H, W)`` datacube. Ragged batches — only
        possible for a scene smaller than one tile — fall back to per-tile.
        """
        arrs = [np.asarray(image, dtype=np.float32) for image in images]
        for arr in arrs:
            if arr.ndim != 3:
                raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
        if len({arr.shape for arr in arrs}) != 1:
            return [self.embed_tile(arr, meta) for arr in arrs]
        self._ensure_model()
        batch = np.stack(arrs, axis=0)
        tokens = self._encode_patches_batch(batch, meta)  # (B, n_tokens, D)
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
                "PyTorch is required for the Clay backend; install the 'inference' dependency tier"
            ) from exc
        self._device = torch.device(
            self._device_pref
            if self._device_pref
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = self._load_model(self._resolve_weights())

    def _resolve_weights(self) -> Path:
        from ._weights import hf_download, resolve_local

        local = resolve_local(self._weights)
        if local is not None:
            return local
        # Signal the start of the (possibly long) first-run download so a UI can
        # show activity; huggingface_hub renders its own byte progress. The
        # recorded SHA-256 (if any) is verified once the file resolves (H-8).
        if self._progress is not None:
            self._progress(0, None)
        return hf_download(
            CLAY_REPO,
            CLAY_CHECKPOINT,
            cache_dir=self._cache_dir,
            expected_sha256=backend_info(self.backend_id).weights_sha256,
        )

    def _metadata_path(self) -> Path:
        """Absolute path to the vendored Clay ``metadata.yaml`` (or raise).

        The checkpoint's saved ``metadata_path`` is relative to the Clay source
        tree, so we override it with this vendored copy. If the asset is missing
        (a broken install), fail with an actionable error rather than letting the
        checkpoint fall back to its unresolvable relative path.
        """
        from pathlib import Path

        path = Path(__file__).resolve().parent.joinpath(*CLAY_METADATA_ASSET)
        if not path.is_file():
            raise RuntimeError(
                "The vendored Clay metadata file is missing from the plugin "
                f"({path}). Reinstall the plugin ZIP — the Clay backend cannot "
                "run without it, because the checkpoint's saved metadata path is "
                "relative to the Clay source tree and does not resolve inside QGIS."
            )
        return path

    def _load_model(self, checkpoint: Path) -> Any:
        """Load the Clay Lightning module from a checkpoint.

        VERIFY during smoke test: the import path / class name below matches the
        installed ``claymodel`` version. Kept isolated so a version bump is a
        one-line change.

        Three checkpoint hyper-parameters are overridden at load time (Lightning
        merges kwargs over the saved ``hyper_parameters``), matching Clay's
        official embedding tutorial:

        * ``metadata_path`` — the checkpoint saved the *relative* default
          ``"configs/metadata.yaml"``, which ``open()`` resolves against the
          current working directory. Inside QGIS the CWD is not the Clay repo, so
          the load crashes with ``FileNotFoundError``. We point it at the metadata
          file vendored beside this module (absolute path).
        * ``mask_ratio=0`` — Clay's encoder unconditionally masks patches; the
          training value (0.75) would drop ~75% of tokens and the grid reshape
          would then fail. Zero keeps every patch.
        * ``shuffle=False`` — the training checkpoint saved ``shuffle=True``, which
          returns patches in a random permutation and scrambles the spatial grid.
          Identity ordering is required for a coherent embedding raster.
        """
        metadata_path = self._metadata_path()
        try:
            from claymodel.module import ClayMAEModule  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                optional_import_error(
                    exc,
                    label="Clay's 'claymodel' package",
                    module="claymodel",
                    install_cmd="pip install git+https://github.com/Clay-foundation/model",
                )
            ) from exc
        # Fail loud with an actionable message if the checkpoint-load call drifts in
        # the installed claymodel (H-6), mirroring the Prithvi/Satlas guards, rather
        # than letting an opaque error escape at a user's first embed run.
        try:
            model = ClayMAEModule.load_from_checkpoint(
                str(checkpoint),
                map_location=self._device,
                metadata_path=str(metadata_path),
                mask_ratio=0.0,
                shuffle=False,
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise unsupported_model_error(
                "clay",
                f"ClayMAEModule.load_from_checkpoint failed ({exc})",
            ) from exc
        model.eval()
        model.to(self._device)
        # Inference is wrapped in ``torch.no_grad()`` at the call site; we do not
        # call ``torch.set_grad_enabled(False)`` here because that would disable
        # autograd process-wide and could break other torch-using QGIS plugins.
        return model

    def _encoder(self) -> Any:
        """Return the Clay encoder module, or raise an actionable error (H-6).

        This is the one uncertain external-API seam (``.model.encoder``, with a
        fallback to the model itself). Isolated and torch-free so the guard is
        checked in a single, unit-testable place; a version bump is a one-line
        change here.
        """
        inner = getattr(self._model, "model", self._model)
        return require_model_attr(
            inner,
            "encoder",
            "clay",
            "the loaded model exposes no '.model.encoder' attribute",
        )

    def _encode_patches(self, arr: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        """Run the Clay encoder, returning ``(n_tokens, D)`` patch tokens."""
        import torch

        datacube = self._build_datacube(arr, meta)
        encoder = self._encoder()
        with torch.no_grad():
            output = encoder(datacube)
        tokens = select_feature(output, "clay", index=0)
        tensor = tokens.detach().to("cpu").float()
        if tensor.ndim == 3:  # (B, n_tokens, D) -> drop batch
            tensor = tensor[0]
        result: NDArray[Any] = tensor.numpy()
        return result

    def _encode_patches_batch(self, batch: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        """Run the Clay encoder over a ``(B, C, H, W)`` batch -> ``(B, n_tokens, D)``.

        VERIFY during smoke test alongside :meth:`_encode_patches`: the encoder
        must accept a batched datacube (it does in Clay v1.5); this seam is kept
        isolated so a version bump is a one-line change.
        """
        import torch

        datacube = self._build_datacube_batch(batch, meta)
        encoder = self._encoder()
        with torch.no_grad():
            output = encoder(datacube)
        tokens = select_feature(output, "clay", index=0)
        tensor = tokens.detach().to("cpu").float()
        if tensor.ndim == 2:  # (n_tokens, D) -> single-sample batch
            tensor = tensor.unsqueeze(0)
        result: NDArray[Any] = tensor.numpy()
        return result

    def _sensor_stats(self) -> dict[str, dict[str, float]]:
        """Per-band mean/std for the default sensor, from the vendored metadata.

        Parsed once and cached. ``yaml`` is imported lazily here (not at module
        import) so this module stays import-light; at embed time the Clay package —
        which itself depends on PyYAML — is installed, so the import always
        succeeds on the path that reaches here.
        """
        if self._stats_cache is None:
            import yaml  # type: ignore[import-untyped]

            with self._metadata_path().open(encoding="utf-8") as handle:
                meta = yaml.safe_load(handle)
            bands = (meta.get(CLAY_METADATA_PLATFORM) or {}).get("bands") or {}
            self._stats_cache = {
                "mean": dict(bands.get("mean") or {}),
                "std": dict(bands.get("std") or {}),
                "wavelength": dict(bands.get("wavelength") or {}),
            }
        return self._stats_cache

    def _band_stats(self, bands: Sequence[str]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return ``(means, stds)`` aligned to ``bands`` as ``(C, 1, 1)`` arrays.

        Each semantic band name is mapped to its Sentinel-2 metadata key
        (:data:`_S2_METADATA_KEY`), then looked up directly by name; a band with no
        published statistic falls back to identity (mean 0, std 1) — passed through
        unstandardised. Clay's published stats cover the Sentinel-2 bands the plugin
        fetches by default, so the common path always resolves.
        """
        stats = self._sensor_stats()
        means: list[float] = []
        stds: list[float] = []
        for band in bands:
            key = _S2_METADATA_KEY.get(band, band)
            mean = stats["mean"].get(key)
            std = stats["std"].get(key)
            if mean is None or std is None or float(std) == 0.0:
                means.append(0.0)
                stds.append(1.0)
            else:
                means.append(float(mean))
                stds.append(float(std))
        means_arr = np.asarray(means, dtype=np.float32).reshape(-1, 1, 1)
        stds_arr = np.asarray(stds, dtype=np.float32).reshape(-1, 1, 1)
        return means_arr, stds_arr

    def _standardize(self, arr: NDArray[Any], bands: Sequence[str]) -> NDArray[Any]:
        """Clay's expected input: per-band ``(DN - mean) / std`` (see module docstring).

        Matches Clay's official embedding tutorial, which feeds raw
        surface-reflectance DN through ``torchvision Normalize(mean, std)`` with the
        per-sensor statistics in ``configs/metadata.yaml``. ``reflectance_scale`` is
        an optional pre-scale into those DN units (default 1.0). The ``(C, 1, 1)``
        stats broadcast over both a single ``(C, H, W)`` tile and a ``(B, C, H, W)``
        batch.
        """
        means, stds = self._band_stats(bands)
        scaled = np.asarray(arr, dtype=np.float32) * self._scale
        standardized: NDArray[Any] = ((scaled - means) / stds).astype(np.float32, copy=False)
        return standardized

    def _band_waves(self, bands: Sequence[str]) -> list[float]:
        """Central wavelength (micrometres) per band, from the vendored metadata.

        Read from the ``wavelength`` block of the metadata (keyed like mean/std),
        mapping semantic names via :data:`_S2_METADATA_KEY`. Falls back to the
        module :data:`WAVELENGTHS_UM` table, then to ``0.0`` for a band with no
        published wavelength — so the common Sentinel-2 path always resolves from
        the metadata and a custom band never raises. Keeping the values in the
        vendored metadata (rather than hard-coded here) is the single source of
        truth Clay itself uses.
        """
        published = self._sensor_stats()["wavelength"]
        waves: list[float] = []
        for band in bands:
            key = _S2_METADATA_KEY.get(band, band)
            value = published.get(key)
            if value is None:
                value = WAVELENGTHS_UM.get(band)
            waves.append(float(value) if value is not None else 0.0)
        return waves

    def _check_token_count(self, n_tokens: int, grid_h: int, grid_w: int) -> None:
        """Guard the encoder's token count against the expected patch grid (H-6).

        ``tokens_to_grid`` drops leading class/register tokens, so a correct run
        has ``n_tokens`` equal to ``grid_h * grid_w`` plus a small number of
        leading tokens. A negative or large excess means the configured patch
        size disagrees with the checkpoint (or the encoder output changed shape);
        without this guard the grid would be silently mis-sliced and scrambled.
        """
        expected = grid_h * grid_w
        extra = n_tokens - expected
        if extra < 0 or extra > _MAX_NON_PATCH_TOKENS:
            raise unsupported_model_error(
                "clay",
                f"the encoder returned {n_tokens} tokens but a {grid_h}x{grid_w} "
                f"patch grid expects {expected} (plus at most "
                f"{_MAX_NON_PATCH_TOKENS} leading class/register tokens); the "
                f"configured patch size ({self._patch_size}) likely disagrees "
                "with the checkpoint",
            )

    def _build_datacube(self, arr: NDArray[Any], meta: TileMeta) -> dict[str, Any]:
        import torch

        pixels = torch.from_numpy(self._standardize(arr, meta.bands)).unsqueeze(0).to(self._device)
        waves = torch.tensor(
            self._band_waves(meta.bands),
            dtype=torch.float32,
            device=self._device,
        )
        return {
            "pixels": pixels,
            "time": self._time_encoding(meta.datetime).to(self._device),
            "latlon": self._latlon_encoding(meta.lat, meta.lon).to(self._device),
            "gsd": torch.tensor(float(meta.gsd), device=self._device),
            "waves": waves,
        }

    def _build_datacube_batch(self, batch: NDArray[Any], meta: TileMeta) -> dict[str, Any]:
        """Datacube for a ``(B, C, H, W)`` batch; time/latlon repeated per sample."""
        import torch

        count = int(batch.shape[0])
        pixels = torch.from_numpy(self._standardize(batch, meta.bands)).to(self._device)
        waves = torch.tensor(
            self._band_waves(meta.bands),
            dtype=torch.float32,
            device=self._device,
        )
        time = self._time_encoding(meta.datetime).to(self._device).repeat(count, 1)
        latlon = self._latlon_encoding(meta.lat, meta.lon).to(self._device).repeat(count, 1)
        return {
            "pixels": pixels,
            "time": time,
            "latlon": latlon,
            "gsd": torch.tensor(float(meta.gsd), device=self._device),
            "waves": waves,
        }

    @staticmethod
    def _time_encoding(datetime_iso: str | None) -> Any:
        import torch

        week, hour = 0.0, 0.0
        if datetime_iso:
            import datetime as _dt

            try:
                stamp = _dt.datetime.fromisoformat(datetime_iso.replace("Z", "+00:00"))
                week = stamp.isocalendar().week / 52.0
                hour = stamp.hour / 24.0
            except ValueError:
                pass
        vec = [
            math.sin(2 * math.pi * week),
            math.cos(2 * math.pi * week),
            math.sin(2 * math.pi * hour),
            math.cos(2 * math.pi * hour),
        ]
        return torch.tensor([vec], dtype=torch.float32)

    @staticmethod
    def _latlon_encoding(lat: float | None, lon: float | None) -> Any:
        import torch

        lat_r = math.radians(lat) if lat is not None else 0.0
        lon_r = math.radians(lon) if lon is not None else 0.0
        vec = [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)]
        return torch.tensor([vec], dtype=torch.float32)
