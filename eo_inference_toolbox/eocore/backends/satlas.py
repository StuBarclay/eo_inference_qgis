"""SatlasPretrain backend: Swin-v2 (or ResNet) + FPN, via satlaspretrain-models.

Unlike the ViT backends, Satlas emits several feature maps at different scales
(FPN outputs). This backend resamples each to the tile grid and concatenates
them into a single per-pixel embedding vector, so it presents the same
``(D, h, w)`` interface as the others.

Torch and satlaspretrain-models are imported lazily inside methods. By default
weights are downloaded automatically by the package from its checkpoint id (e.g.
``Sentinel2_SwinB_SI_RGB``); alternatively a local SatlasPretrain-format
checkpoint (a torch state dict) can be supplied, in which case the model
architecture is looked up from the checkpoint id and the local weights are loaded
into it — no download.

Notes on input scaling
----------------------
The Sentinel-2 *RGB* checkpoints expect a true-colour image (the B4/B3/B2 bands,
in that order) with 8-bit 0-255 values divided by 255, as documented upstream.
The default ``input_scale`` (1/255) matches that convention; for raw Sentinel-2
L2A reflectance (0-10000) pass a scale that maps your data into roughly
``[0, 1]`` (see docs/USER_GUIDE.md and docs/LIMITATIONS.md).

Model coverage
--------------
This backend feeds a single ``(B, C, H, W)`` image, so it targets the
*single-image* RGB checkpoints (default ``Sentinel2_SwinB_SI_RGB``). The
multi-image (``..._MI_...``) checkpoints expect several aligned images stacked
on the channel axis and are not the intended input for this code path.

References
----------
- https://github.com/allenai/satlaspretrain_models (ODC-BY weights).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..descriptor import EmbeddingDescriptor
from ._deps import optional_import_error
from ._pool import fuse_feature_maps
from .base import (
    backend_info,
    descriptor_from_info,
    require_model_attr,
    unsupported_model_error,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._weights import ProgressFn
    from .base import TileMeta


class SatlasBackend:
    """SatlasPretrain embedding backend (FPN feature maps pooled per pixel)."""

    backend_id = "satlas"
    display_name = "SatlasPretrain (Allen Institute / AI2)"

    def __init__(
        self,
        *,
        checkpoint: str = "Sentinel2_SwinB_SI_RGB",
        weights: str | Path | None = None,
        device: str | None = None,
        input_scale: float = 1.0 / 255.0,
        fuse: str = "concat",
        upsampling: str = "nearest",
        progress: ProgressFn | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        # A local checkpoint (a SatlasPretrain-format state dict) is loaded into a
        # freshly-built model whose architecture is looked up from the checkpoint
        # id; ``None`` fetches the pretrained weights by id (see ``_load_model``).
        self._weights = weights
        self._device_pref = device
        self._scale = float(input_scale)
        self._fuse = fuse
        self._upsampling = upsampling
        self._progress = progress
        self._model: Any = None
        self._device: Any = None
        self._embedding_dim: int | None = None
        # The checkpoint's expected input channel count, captured from the weights
        # catalogue at load time and used to guard the tile band count.
        self._expected_channels: int | None = None

    # -- interface ---------------------------------------------------------
    def descriptor(self) -> EmbeddingDescriptor:
        return descriptor_from_info(backend_info(self.backend_id), n_bands=self._embedding_dim)

    def required_bands(self) -> tuple[str, ...]:
        return backend_info(self.backend_id).default_bands

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
        n_bands, height, width = arr.shape
        self._ensure_model()
        self._guard_channels(n_bands)
        maps = self._encode_maps(arr)  # list of (C_i, h_i, w_i)
        fused = fuse_feature_maps(maps, height, width, mode=self._fuse, method=self._upsampling)
        self._embedding_dim = int(fused.shape[0])
        return fused

    def embed_tiles(self, images: Sequence[NDArray[Any]], meta: TileMeta) -> list[NDArray[Any]]:
        """Embed a batch of equal-shaped tiles through the FPN in one pass (E-2).

        Satlas ignores ``meta``; equal-shaped tiles stack into one
        ``(B, C, H, W)`` input and the per-scale feature maps are split back per
        sample. Ragged batches fall back to per-tile.
        """
        arrs = [np.asarray(image, dtype=np.float32) for image in images]
        for arr in arrs:
            if arr.ndim != 3:
                raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
        if len({arr.shape for arr in arrs}) != 1:
            return [self.embed_tile(arr, meta) for arr in arrs]
        self._ensure_model()
        self._guard_channels(int(arrs[0].shape[0]))
        batch = np.stack(arrs, axis=0)
        per_sample_maps = self._encode_maps_batch(batch)
        _, height, width = arrs[0].shape
        outputs: list[NDArray[Any]] = []
        for maps in per_sample_maps:
            fused = fuse_feature_maps(maps, height, width, mode=self._fuse, method=self._upsampling)
            self._embedding_dim = int(fused.shape[0])
            outputs.append(fused)
        return outputs

    # -- model plumbing ----------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyTorch is required for the Satlas backend; install the "
                "'inference' dependency tier"
            ) from exc
        self._device = torch.device(
            self._device_pref
            if self._device_pref
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = self._load_model()

    def _load_model(self) -> Any:
        """Load a Satlas model with its FPN, from a download or a local checkpoint.

        With no local weights the pretrained checkpoint is fetched by id via
        ``Weights().get_pretrained_model``; with a local checkpoint the model
        architecture is looked up from the checkpoint id and the local state dict
        is loaded into a freshly-built ``Model`` (no download).

        VERIFY during smoke test: the ``Weights().get_pretrained_model`` /
        ``Model`` signatures and the ``SatlasPretrain_weights`` catalogue match the
        installed satlaspretrain-models version.
        """
        try:
            import satlaspretrain_models  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                optional_import_error(
                    exc,
                    label="the 'satlaspretrain_models' package (needed for Satlas)",
                    module="satlaspretrain_models",
                    install_cmd="pip install satlaspretrain-models",
                )
            ) from exc
        # Record the checkpoint's expected input channel count (for the band-count
        # guard) from the catalogue when it is available; an unknown id leaves it
        # None so the guard is simply skipped.
        catalogue = self._weights_catalogue(satlaspretrain_models)
        if catalogue is not None and self._checkpoint in catalogue:
            entry = catalogue[self._checkpoint]
            channels = entry.get("num_channels") if isinstance(entry, Mapping) else None
            self._expected_channels = int(channels) if channels is not None else None
        if self._weights is not None:
            model = self._load_local_model(satlaspretrain_models, catalogue)
        else:
            model = self._download_model(satlaspretrain_models)
        if not callable(model):
            raise unsupported_model_error(
                "satlas", f"the model for {self._checkpoint!r} is not callable"
            )
        model.eval()
        model.to(self._device)
        # Inference is wrapped in ``torch.no_grad()`` at the call site; we avoid
        # ``torch.set_grad_enabled(False)`` here (it is a process-wide side effect).
        return model

    def _download_model(self, satlaspretrain_models: Any) -> Any:
        """Fetch the pretrained checkpoint by id via ``Weights().get_pretrained_model``."""
        # Signal the start of the (possibly long) first-run weight download that
        # satlaspretrain-models performs internally, so a UI can show activity.
        if self._progress is not None:
            self._progress(0, None)
        manager = satlaspretrain_models.Weights()
        # Fail loud with an actionable message if the installed
        # satlaspretrain-models has renamed/dropped the entry point (H-6).
        get_pretrained_model = require_model_attr(
            manager,
            "get_pretrained_model",
            "satlas",
            "satlaspretrain_models.Weights has no 'get_pretrained_model' method",
        )
        # Fail loud with an actionable message if the build call itself drifts in
        # the installed satlaspretrain-models (H-6), mirroring the Prithvi guard,
        # rather than letting an opaque error escape at a user's first embed run.
        try:
            return get_pretrained_model(self._checkpoint, fpn=True)
        except (AttributeError, KeyError, TypeError) as exc:
            raise unsupported_model_error(
                "satlas",
                f"Weights.get_pretrained_model({self._checkpoint!r}) failed ({exc})",
            ) from exc

    def _load_local_model(
        self, satlaspretrain_models: Any, catalogue: Mapping[str, Any] | None
    ) -> Any:
        """Build the model for the checkpoint id and load a local state dict into it.

        The local file must be a SatlasPretrain-format checkpoint (a torch state
        dict, as published on Hugging Face) matching the ``checkpoint`` id's
        architecture, which supplies the channel count, backbone and single/multi
        -image flag — mirroring what ``get_pretrained_model`` does after its
        download.
        """
        import torch

        from ._weights import resolve_local

        # Clear FileNotFoundError up front if the path is missing.
        path = resolve_local(self._weights)
        if path is None:  # pragma: no cover - guarded by the caller (weights is not None)
            raise unsupported_model_error("satlas", "no local weights path was provided")
        num_channels, multi_image, backbone = self._lookup_architecture(catalogue, self._checkpoint)
        if self._progress is not None:
            # Loading a local checkpoint is usually quick, but announce activity.
            self._progress(0, None)
        try:
            state_dict = torch.load(str(path), map_location=self._device)
        # A corrupt / non-Satlas file can fail torch.load in many ways (pickle,
        # zip, UnpicklingError, RuntimeError, ...); catch broadly and re-raise as
        # one actionable message rather than leak a deserialize traceback.
        except Exception as exc:
            raise unsupported_model_error(
                "satlas",
                f"could not read the local Satlas weights file {path} ({exc}); it must be a "
                "SatlasPretrain-format checkpoint (a torch state dict)",
            ) from exc
        model_cls = require_model_attr(
            satlaspretrain_models,
            "Model",
            "satlas",
            "satlaspretrain_models has no 'Model' class",
        )
        try:
            return model_cls(num_channels, multi_image, backbone, fpn=True, weights=state_dict)
        except (AttributeError, KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise unsupported_model_error(
                "satlas",
                f"building the Satlas model for local checkpoint {path.name} failed ({exc}); "
                f"the file must be a SatlasPretrain-format state dict matching the "
                f"{self._checkpoint!r} architecture ({num_channels}-channel, backbone {backbone})",
            ) from exc

    @staticmethod
    def _weights_catalogue(satlaspretrain_models: Any) -> Mapping[str, Any] | None:
        """The ``SatlasPretrain_weights`` id->architecture catalogue, or ``None`` if absent.

        Returned as ``None`` (rather than raising) when the installed package does
        not expose the catalogue, so the download path and channel guard degrade
        gracefully; the local path turns a missing catalogue into a clear error.
        """
        try:
            from satlaspretrain_models.utils import (  # type: ignore[import-not-found]
                SatlasPretrain_weights,
            )
        except ImportError:
            return None
        return SatlasPretrain_weights if isinstance(SatlasPretrain_weights, Mapping) else None

    @staticmethod
    def _lookup_architecture(
        catalogue: Mapping[str, Any] | None, checkpoint: str
    ) -> tuple[int, bool, Any]:
        """Resolve ``(num_channels, multi_image, backbone)`` for a checkpoint id.

        Pure logic over the ``SatlasPretrain_weights`` catalogue — no torch — so it
        is unit-testable with a fake catalogue. Raises an actionable error when the
        catalogue is unavailable, the id is unknown (its architecture cannot be
        inferred), or the entry is missing fields.
        """
        if catalogue is None:
            raise unsupported_model_error(
                "satlas",
                "the installed satlaspretrain-models does not expose the "
                "'SatlasPretrain_weights' catalogue, so a local checkpoint's architecture "
                "cannot be resolved; pin a known-good satlaspretrain-models version",
            )
        if checkpoint not in catalogue:
            known = ", ".join(sorted(catalogue)[:6])
            raise unsupported_model_error(
                "satlas",
                f"checkpoint id {checkpoint!r} is not in the SatlasPretrain catalogue, so the "
                "architecture for your local weights is unknown; set the checkpoint id to the "
                f"one whose architecture your file matches (e.g. 'Sentinel2_SwinB_SI_RGB'). "
                f"Known ids include: {known}",
            )
        entry = catalogue[checkpoint]
        try:
            return int(entry["num_channels"]), bool(entry["multi_image"]), entry["backbone"]
        except (KeyError, TypeError) as exc:
            raise unsupported_model_error(
                "satlas",
                f"the SatlasPretrain catalogue entry for {checkpoint!r} is missing an "
                f"architecture field (num_channels / multi_image / backbone): {exc}",
            ) from exc

    @staticmethod
    def _channel_count_error(got: int, expected: int | None, checkpoint: str) -> str | None:
        """Message if a tile's band count disagrees with the checkpoint, else ``None``."""
        if expected is None or got == expected:
            return None
        return (
            f"the Satlas checkpoint {checkpoint!r} expects {expected}-channel input but the "
            f"tile has {got} band(s); set the band-order override to the checkpoint's bands or "
            "select the checkpoint id whose channel count matches your imagery"
        )

    def _guard_channels(self, n_bands: int) -> None:
        """Raise if a tile's band count disagrees with the loaded checkpoint (H-6).

        The expected count comes from the ``SatlasPretrain_weights`` catalogue at
        load time; when it is unknown (``None``) the guard is a no-op. Mirrors the
        explicit band-count guard in the Prithvi backend so a channel mismatch
        fails with an actionable message before the opaque model error.
        """
        message = self._channel_count_error(n_bands, self._expected_channels, self._checkpoint)
        if message is not None:
            raise ValueError(message)

    @staticmethod
    def _as_feature_map_sequence(output: Any) -> list[Any]:
        """Normalise a headless backbone/FPN forward output to a list of maps.

        A SatlasPretrain model loaded **without a prediction head** — which is all
        this backend ever does — returns the multi-scale feature maps as a plain
        sequence (a ``list`` per the satlaspretrain README). The ``(outputs, loss)``
        tuple shape only arises when a head is attached, so it cannot occur here.
        Therefore *any* non-empty ``list``/``tuple`` is the full feature-map
        sequence and must be kept whole: slicing ``output[0]`` would silently keep
        only the finest scale and discard every other FPN level (H-6). A bare single
        tensor is wrapped; an empty sequence fails loud.
        """
        if isinstance(output, (list, tuple)):
            if not output:
                raise unsupported_model_error("satlas", "the model returned no feature maps")
            return list(output)
        return [output]

    @staticmethod
    def _split_maps_per_sample(
        map_arrays: Sequence[NDArray[Any]], count: int
    ) -> list[list[NDArray[Any]]]:
        """Split each batched ``(B, C, h, w)`` map array into per-sample maps.

        Guards that every feature map is batch-first with a leading dimension equal
        to the batch size *before* indexing it. A map that is not batch-first (e.g.
        an unexpectedly pooled / global map, or a drifted API) would otherwise be
        silently mis-sliced along its channel axis, scrambling which channels belong
        to which tile; instead it fails loud (H-6).
        """
        per_sample: list[list[NDArray[Any]]] = [[] for _ in range(count)]
        for array in map_arrays:
            if array.ndim < 1 or int(array.shape[0]) != count:
                raise unsupported_model_error(
                    "satlas",
                    f"a feature map with shape {tuple(array.shape)} does not carry the "
                    f"expected batch size {count} as its leading dimension, so a batched "
                    "forward pass cannot be split back per tile",
                )
            for index in range(count):
                per_sample[index].append(np.asarray(array[index]))
        return per_sample

    def _encode_maps(self, arr: NDArray[Any]) -> list[NDArray[Any]]:
        """Forward pass; returns a list of ``(C_i, h_i, w_i)`` feature maps."""
        import torch

        tensor = torch.from_numpy(arr * self._scale).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        feature_maps = self._as_feature_map_sequence(output)
        maps: list[NDArray[Any]] = []
        for fmap in feature_maps:
            array = fmap.detach().to("cpu").float()
            if array.ndim == 4:  # (B, C, h, w) -> drop batch
                array = array[0]
            maps.append(array.numpy())
        return maps

    def _encode_maps_batch(self, batch: NDArray[Any]) -> list[list[NDArray[Any]]]:
        """Forward pass over a ``(B, C, H, W)`` batch; feature maps split per sample.

        Returns a list (one entry per input tile) of that tile's ``(C_i, h_i, w_i)``
        feature maps. VERIFY during smoke test alongside :meth:`_encode_maps`.
        """
        import torch

        tensor = torch.from_numpy(batch * self._scale).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        feature_maps = self._as_feature_map_sequence(output)
        map_arrays = [fmap.detach().to("cpu").float().numpy() for fmap in feature_maps]
        return self._split_maps_per_sample(map_arrays, int(batch.shape[0]))
