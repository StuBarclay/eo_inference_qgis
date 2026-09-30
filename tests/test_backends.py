"""Tests for the torch-free backend registry (facts, not model loading)."""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eo_inference_toolbox.eocore.backends import (
    BACKEND_IDS,
    BackendInfo,
    BatchEmbeddingBackend,
    EmbeddingBackend,
    TileMeta,
    backend_info,
    band_requirement_error,
    embed_tile_batch,
    load_backend,
    scale_sanity_warning,
)
from eo_inference_toolbox.eocore.backends.base import (
    descriptor_from_info,
    resolve_bands,
)


def test_all_three_backends_registered() -> None:
    assert set(BACKEND_IDS) == {"clay", "prithvi", "satlas"}


def test_backend_info_is_complete() -> None:
    for backend_id in BACKEND_IDS:
        info = backend_info(backend_id)
        assert isinstance(info, BackendInfo)
        assert info.display_name
        assert info.summary
        assert info.default_bands
        assert info.embedding_dim > 0
        assert info.resolution_m > 0
        assert "torch" in info.pip_packages
        assert info.import_names  # every backend names its import(s)


def test_import_names_match_installed_packages() -> None:
    # Regression: a backend must only declare import names its pip package
    # actually provides. Clay once listed a non-existent top-level ``clay``
    # module alongside ``claymodel``; because the status badge and
    # missing_import_names require *all* names to import, that kept Clay flagged
    # "missing" even after a correct install. The runtime import is
    # ``from claymodel.module import ClayMAEModule`` — only ``claymodel`` exists.
    assert backend_info("clay").import_names == ("claymodel",)
    assert backend_info("prithvi").import_names == ("terratorch",)
    assert backend_info("satlas").import_names == ("satlaspretrain_models",)


def test_clay_checkpoint_path_includes_subfolder() -> None:
    # Regression: the Clay v1.5 checkpoint is stored at ``v1.5/clay-v1.5.ckpt`` in
    # the Hugging Face repo, not at the root. A bare "clay-v1.5.ckpt" 404s, which
    # would break BOTH the wizard's Download/verify-weights action and the first
    # Embed run (its lazy weight fetch uses the same constant).
    from eo_inference_toolbox.eocore.backends.clay import CLAY_CHECKPOINT, CLAY_REPO

    assert CLAY_REPO == "made-with-clay/Clay"
    assert CLAY_CHECKPOINT == "v1.5/clay-v1.5.ckpt"


def test_clay_metadata_asset_is_vendored_and_absolute() -> None:
    # Regression: ``ClayMAEModule.load_from_checkpoint`` re-instantiates the module
    # from the checkpoint's saved hyper-parameters, one of which is a *relative*
    # ``metadata_path`` ("configs/metadata.yaml"). Inside QGIS the CWD is not the
    # Clay repo, so the load crashed with FileNotFoundError. We vendor the metadata
    # beside the backend and pass its absolute path as an override. This test needs
    # neither torch nor claymodel: ``_metadata_path`` only touches the filesystem.
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    path = ClayBackend()._metadata_path()
    assert path.is_absolute()
    assert path.is_file()
    assert path.name == "metadata.yaml"


def test_clay_metadata_asset_parses_with_sentinel2_stats() -> None:
    # The vendored file must be valid YAML and carry the per-band mean / std /
    # wavelength that Clay's DynamicEmbedding needs for sentinel-2-l2a.
    yaml = pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    with ClayBackend()._metadata_path().open() as handle:
        meta = yaml.safe_load(handle)

    assert "sentinel-2-l2a" in meta
    s2 = meta["sentinel-2-l2a"]
    assert s2["band_order"][0] == "blue"
    assert s2["gsd"] == 10
    for stat in ("mean", "std", "wavelength"):
        assert "blue" in s2["bands"][stat]
    # Stats are published in raw DN units, so the mean is in the thousands, not ~0.
    assert s2["bands"]["mean"]["blue"] > 100
    assert s2["bands"]["std"]["blue"] > 100
    assert 0.4 < s2["bands"]["wavelength"]["blue"] < 0.6


def test_clay_band_stats_match_sentinel2_metadata() -> None:
    # Clay's embedding path expects per-band (DN - mean) / std with the sensor
    # statistics from configs/metadata.yaml. The plugin's semantic band names map
    # to Sentinel-2 metadata keys (narrow_nir->nir08, swir1->swir16, swir2->swir22);
    # the stats must come back aligned to the requested band order. Torch-free.
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    means, stds = ClayBackend()._band_stats(("red", "green", "blue", "nir"))
    assert means.shape == (4, 1, 1)  # broadcasts over (C, H, W) and (B, C, H, W)
    assert stds.shape == (4, 1, 1)
    # Values are the published Sentinel-2 L2A stats, in the requested order.
    assert means.ravel().tolist() == [1552.0, 1355.0, 1105.0, 2743.0]
    assert stds.ravel().tolist() == [1888.0, 1757.0, 1809.0, 1742.0]


def test_clay_band_stats_maps_renamed_bands() -> None:
    # narrow_nir/swir1/swir2 differ from Clay's keys and must still resolve.
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    means, stds = ClayBackend()._band_stats(("narrow_nir", "swir1", "swir2"))
    assert means.ravel().tolist() == [2785.0, 2388.0, 1835.0]  # nir08/swir16/swir22
    assert stds.ravel().tolist() == [1648.0, 1470.0, 1379.0]


def test_clay_band_stats_unknown_band_falls_back_to_identity() -> None:
    # A band with no published statistic passes through unstandardised (mean 0,
    # std 1) rather than raising, so a custom band never crashes the embed.
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    means, stds = ClayBackend()._band_stats(("not_a_band",))
    assert means.ravel().tolist() == [0.0]
    assert stds.ravel().tolist() == [1.0]


def test_clay_standardize_applies_dn_mean_std() -> None:
    # (DN - mean) / std per band, with the default pre-scale of 1.0 (raw DN in).
    # red: mean 1552, std 1888; blue: mean 1105, std 1809. Torch-free (numpy only).
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    arr = np.array(
        [[[1552.0, 3440.0]], [[1105.0, 2914.0]]],  # (2 bands, 1, 2): red, blue
        dtype=np.float32,
    )
    out = ClayBackend()._standardize(arr, ("red", "blue"))
    assert out.shape == arr.shape
    assert out.dtype == np.float32
    # red row: (1552-1552)/1888=0 ; (3440-1552)/1888=1
    assert out[0].ravel().tolist() == pytest.approx([0.0, 1.0])
    # blue row: (1105-1105)/1809=0 ; (2914-1105)/1809=1
    assert out[1].ravel().tolist() == pytest.approx([0.0, 1.0])


def test_clay_standardize_pre_scale_reaches_dn_units() -> None:
    # reflectance_scale is a pre-scale into DN units: reflectance in [0, 1] with
    # scale 1e4 standardises identically to the raw-DN input above.
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    reflectance = np.array([[[0.1552, 0.3440]]], dtype=np.float32)  # red / 10000
    out = ClayBackend(reflectance_scale=1.0e4)._standardize(reflectance, ("red",))
    assert out.ravel().tolist() == pytest.approx([0.0, 1.0], abs=1e-3)


def test_clay_band_waves_read_from_metadata() -> None:
    # Central wavelengths come from the vendored metadata's ``wavelength`` block
    # (single source of truth), using the same semantic->S2 key mapping as the
    # stats: narrow_nir->nir08, swir1->swir16. Torch-free.
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    waves = ClayBackend()._band_waves(("blue", "narrow_nir", "swir1"))
    assert waves == pytest.approx([0.493, 0.865, 1.61])


def test_clay_band_waves_unknown_band_falls_back_to_zero() -> None:
    # A band with no published wavelength (and none in the fallback table) yields
    # 0.0 rather than raising, mirroring the identity fallback in the stats path.
    pytest.importorskip("yaml")

    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    assert ClayBackend()._band_waves(("not_a_band",)) == [0.0]


def test_clay_token_count_guard_accepts_grid_plus_class_token() -> None:
    # tokens_to_grid drops leading class/register tokens, so an exact grid or a
    # grid + a few leading tokens is fine. No model, no yaml, no torch needed.
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    backend = ClayBackend()
    backend._check_token_count(1024, 32, 32)  # exact patch grid
    backend._check_token_count(1025, 32, 32)  # + one class token


def test_clay_token_count_guard_rejects_mismatch() -> None:
    # A gross mismatch (patch size disagrees with the checkpoint) must fail loud
    # instead of silently mis-slicing and scrambling the grid.
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    backend = ClayBackend()
    with pytest.raises(RuntimeError, match="patch grid expects"):
        backend._check_token_count(4096, 32, 32)  # far too many tokens
    with pytest.raises(RuntimeError, match="patch grid expects"):
        backend._check_token_count(1000, 32, 32)  # fewer tokens than patches


def test_clay_load_from_checkpoint_failure_wraps_as_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the installed claymodel's load_from_checkpoint signature/behaviour has
    # drifted, the opaque TypeError/AttributeError/KeyError must be caught and
    # re-raised as the uniform "cannot use the loaded model" guidance (H-6), not
    # surface raw at a user's first embed. A fake claymodel.module lets us reach
    # the wrap without torch, real weights, or the (uninstalled) claymodel package.
    from eo_inference_toolbox.eocore.backends.clay import ClayBackend

    class _FakeClayMAEModule:
        @staticmethod
        def load_from_checkpoint(*args: Any, **kwargs: Any) -> Any:
            raise TypeError("load_from_checkpoint signature drifted")

    fake_pkg = types.ModuleType("claymodel")
    fake_mod = types.ModuleType("claymodel.module")
    fake_mod.ClayMAEModule = _FakeClayMAEModule  # type: ignore[attr-defined]
    fake_pkg.module = fake_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claymodel", fake_pkg)
    monkeypatch.setitem(sys.modules, "claymodel.module", fake_mod)

    backend = ClayBackend()
    with pytest.raises(RuntimeError, match="cannot use the loaded model"):
        backend._load_model(Path("/tmp/clay-v1.5.ckpt"))


def test_satlas_default_checkpoint_and_bands() -> None:
    # Regression: pin the default SatlasPretrain checkpoint id and the RGB band
    # order. The RGB models input B4/B3/B2 (red, green, blue) in that order, so a
    # silent reorder or a typo'd checkpoint id would degrade embeddings without
    # any error. Constructing the backend is torch-free (torch imports lazily).
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    backend = SatlasBackend()
    assert backend._checkpoint == "Sentinel2_SwinB_SI_RGB"
    assert backend_info("satlas").default_bands == ("red", "green", "blue")


def test_satlas_fetches_the_prerendered_visual_tci_asset() -> None:
    # The default RGB checkpoint expects 8-bit true-colour input, so the fetch
    # pulls Sentinel-2's pre-rendered ``visual`` (TCI) asset in one warp rather
    # than raw reflectance bands (which land at ~36x the model range and only
    # warn). Clay/Prithvi resolve per-band reflectance COGs, so they declare none.
    assert backend_info("satlas").fetch_asset == "visual"
    assert backend_info("clay").fetch_asset is None
    assert backend_info("prithvi").fetch_asset is None


def test_satlas_feature_map_sequence_keeps_all_scales() -> None:
    # A headless Satlas model returns the multi-scale feature maps as a sequence;
    # the whole sequence must be kept. Slicing [0] (the old tuple handling) would
    # silently drop every FPN level but the finest. Torch-free: the helper only
    # inspects the container, not the tensors.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    maps = [np.zeros((1, 128, 8, 8)), np.zeros((1, 256, 4, 4)), np.zeros((1, 512, 2, 2))]
    assert SatlasBackend._as_feature_map_sequence(maps) == maps  # list kept whole
    assert SatlasBackend._as_feature_map_sequence(tuple(maps)) == maps  # tuple too
    single = np.zeros((1, 128, 8, 8))
    wrapped = SatlasBackend._as_feature_map_sequence(single)
    assert len(wrapped) == 1 and wrapped[0] is single  # bare tensor wrapped


def test_satlas_feature_map_sequence_empty_raises() -> None:
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    with pytest.raises(RuntimeError, match="no feature maps"):
        SatlasBackend._as_feature_map_sequence([])


def test_satlas_split_maps_per_sample_batch_first() -> None:
    # Each (B, C, h, w) map splits into per-sample (C, h, w) maps in tile order.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    map_a = np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1)
    map_b = np.arange(2 * 5 * 1 * 1).reshape(2, 5, 1, 1)
    per_sample = SatlasBackend._split_maps_per_sample([map_a, map_b], 2)
    assert len(per_sample) == 2
    assert per_sample[0][0].shape == (3, 1, 1)
    assert per_sample[1][1].shape == (5, 1, 1)
    assert np.array_equal(per_sample[1][0], map_a[1])


def test_satlas_split_maps_per_sample_rejects_wrong_leading_dim() -> None:
    # A feature map whose leading dim is not the batch size (e.g. a pooled/global
    # map, or one without a batch axis) must fail loud rather than be mis-sliced
    # along its channel axis.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    good = np.zeros((2, 4, 1, 1))
    bad = np.zeros((3, 4, 1, 1))  # leading dim 3 != batch 2
    with pytest.raises(RuntimeError, match="expected batch size 2"):
        SatlasBackend._split_maps_per_sample([good, bad], 2)


def test_satlas_accepts_local_weights_checkpoint_at_construction() -> None:
    # HIGH-2: a local SatlasPretrain-format checkpoint IS now supported. The
    # constructor just records the path (torch-free); the file is read lazily at
    # model-load time and its architecture is looked up from the checkpoint id, so
    # construction must NOT raise and must not resolve the channel count yet.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    backend = SatlasBackend(weights="/some/local/model.pth")
    assert str(backend._weights) == "/some/local/model.pth"
    assert backend._expected_channels is None  # only resolved when the model loads


# A stand-in for satlaspretrain_models.utils.SatlasPretrain_weights: checkpoint id
# -> architecture (num_channels / multi_image / backbone). Lets the pure lookup and
# channel-guard logic be tested without torch or the (uninstalled) real package.
_FAKE_SATLAS_CATALOGUE = {
    "Sentinel2_SwinB_SI_RGB": {"num_channels": 3, "multi_image": False, "backbone": "SWINB"},
    "Sentinel2_SwinB_SI_MS": {"num_channels": 9, "multi_image": False, "backbone": "SWINB"},
    "Sentinel1_SwinB_SI": {"num_channels": 2, "multi_image": False, "backbone": "SWINB"},
}


def test_satlas_lookup_architecture_returns_channels_multi_backbone() -> None:
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    num_channels, multi_image, backbone = SatlasBackend._lookup_architecture(
        _FAKE_SATLAS_CATALOGUE, "Sentinel2_SwinB_SI_RGB"
    )
    assert num_channels == 3
    assert multi_image is False
    assert backbone == "SWINB"
    # The multi-spectral checkpoint's 9 channels come back too (types coerced).
    ms_channels = SatlasBackend._lookup_architecture(
        _FAKE_SATLAS_CATALOGUE, "Sentinel2_SwinB_SI_MS"
    )
    assert ms_channels[0] == 9


def test_satlas_lookup_architecture_missing_catalogue_raises() -> None:
    # No catalogue (older/renamed satlaspretrain-models) -> a local checkpoint's
    # architecture cannot be resolved, so fail loud with guidance rather than
    # guessing an architecture that would silently mis-load the weights.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    with pytest.raises(RuntimeError, match="does not expose"):
        SatlasBackend._lookup_architecture(None, "Sentinel2_SwinB_SI_RGB")


def test_satlas_lookup_architecture_unknown_id_raises() -> None:
    # An unknown checkpoint id can't map to an architecture; the error names known
    # ids so the user can pick the one their file matches.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    with pytest.raises(RuntimeError, match="not in the SatlasPretrain catalogue"):
        SatlasBackend._lookup_architecture(_FAKE_SATLAS_CATALOGUE, "Not_A_Checkpoint")


def test_satlas_lookup_architecture_incomplete_entry_raises() -> None:
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    incomplete = {"Broken": {"multi_image": False, "backbone": "SWINB"}}  # no num_channels
    with pytest.raises(RuntimeError, match="missing an architecture field"):
        SatlasBackend._lookup_architecture(incomplete, "Broken")


def test_satlas_channel_count_error_flags_only_real_mismatch() -> None:
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    # Match, or an unknown expected count (catalogue absent), -> no error.
    assert SatlasBackend._channel_count_error(3, 3, "Sentinel2_SwinB_SI_RGB") is None
    assert SatlasBackend._channel_count_error(4, None, "Sentinel2_SwinB_SI_RGB") is None
    # A genuine mismatch names both counts and the checkpoint id.
    message = SatlasBackend._channel_count_error(4, 3, "Sentinel2_SwinB_SI_RGB")
    assert message is not None
    assert "expects 3-channel" in message
    assert "4 band(s)" in message
    assert "Sentinel2_SwinB_SI_RGB" in message


def test_satlas_get_pretrained_model_failure_wraps_as_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the installed satlaspretrain-models's get_pretrained_model has drifted,
    # the opaque error must be caught and re-raised as the uniform "cannot use the
    # loaded model" guidance (H-6), mirroring the Prithvi/Clay guards. A fake
    # satlaspretrain_models module lets us reach the build wrap without torch or
    # the (uninstalled) real package; _load_model touches self._device only after
    # the build, so bypassing _ensure_model (which sets it) is safe here.
    from eo_inference_toolbox.eocore.backends.satlas import SatlasBackend

    class _FakeWeights:
        def get_pretrained_model(self, *args: Any, **kwargs: Any) -> Any:
            raise TypeError("get_pretrained_model signature drifted")

    fake_pkg = types.ModuleType("satlaspretrain_models")
    fake_pkg.Weights = _FakeWeights  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "satlaspretrain_models", fake_pkg)

    backend = SatlasBackend()
    with pytest.raises(RuntimeError, match="cannot use the loaded model"):
        backend._load_model()


def test_prithvi_model_identifier_and_geometry() -> None:
    # Regression: pin the TerraTorch backbone id, spatial patch size, embedding
    # width and the fixed six-band input order (verified against the model's
    # config.json). A silent drift in any of these -- a renamed registry key, a
    # patch-size/grid mismatch, a reordered band list -- would scramble
    # embeddings without raising. Constructing the backend is torch-free
    # (torch/terratorch import lazily at model-load time).
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend
    from eo_inference_toolbox.eocore.stac import PRITHVI_BANDS

    backend = PrithviBackend()
    assert backend._model_id == "prithvi_eo_v2_300"
    assert backend._patch_size == 16
    assert backend.required_bands() == PRITHVI_BANDS
    # config.json band order B02,B03,B04,B05,B06,B07 = Blue,Green,Red,nNIR,SWIR1,SWIR2.
    assert PRITHVI_BANDS == ("blue", "green", "red", "narrow_nir", "swir1", "swir2")
    assert backend_info("prithvi").embedding_dim == 1024


def test_prithvi_band_stats_match_published_config() -> None:
    # Prithvi's embedding path expects per-band (DN - mean) / std with the HLS
    # statistics published in the model's config.json, in PRITHVI_BANDS order.
    # The stats must come back aligned to the requested band order. Torch-free.
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend
    from eo_inference_toolbox.eocore.stac import PRITHVI_BANDS

    means, stds = PrithviBackend()._band_stats(PRITHVI_BANDS)
    assert means.shape == (6, 1, 1)  # broadcasts over (C, H, W) and (B, C, H, W)
    assert stds.shape == (6, 1, 1)
    # Published config.json mean / std, in raw HLS surface-reflectance DN units.
    assert means.ravel().tolist() == [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    assert stds.ravel().tolist() == [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


def test_prithvi_band_stats_unknown_band_falls_back_to_identity() -> None:
    # A band with no published statistic passes through unstandardised (mean 0,
    # std 1) rather than raising, mirroring the Clay backend.
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend

    means, stds = PrithviBackend()._band_stats(("not_a_band",))
    assert means.ravel().tolist() == [0.0]
    assert stds.ravel().tolist() == [1.0]


def test_prithvi_standardize_applies_dn_mean_std() -> None:
    # (DN - mean) / std per band over the fixed six-band input, with the default
    # pre-scale of 1.0 (raw DN in). Each band's two pixels are mean and mean+std,
    # so every band standardises to [0, 1]. Torch-free (numpy only).
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend

    # (6 bands, 1, 2): [mean, mean+std] per band in PRITHVI_BANDS order.
    arr = np.array(
        [
            [[1087.0, 1087.0 + 2248.0]],  # blue
            [[1342.0, 1342.0 + 2179.0]],  # green
            [[1433.0, 1433.0 + 2178.0]],  # red
            [[2734.0, 2734.0 + 1850.0]],  # narrow_nir
            [[1958.0, 1958.0 + 1242.0]],  # swir1
            [[1363.0, 1363.0 + 1049.0]],  # swir2
        ],
        dtype=np.float32,
    )
    out = PrithviBackend()._standardize(arr)
    assert out.shape == arr.shape
    assert out.dtype == np.float32
    for band in range(6):
        assert out[band].ravel().tolist() == pytest.approx([0.0, 1.0])


def test_prithvi_standardize_pre_scale_reaches_dn_units() -> None:
    # reflectance_scale is a pre-scale into DN units: reflectance in [0, 1] with
    # scale 1e4 standardises identically to the raw-DN input. Red band only
    # (mean 1433, std 2178): reflectance 0.1433 -> 0 ; (1433+2178)/1e4 -> 1.
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend

    means = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
    stds = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)
    dn = np.stack([means, means + stds], axis=1)[:, np.newaxis, :]  # (6, 1, 2) DN
    reflectance = (dn / 1.0e4).astype(np.float32)
    out = PrithviBackend(reflectance_scale=1.0e4)._standardize(reflectance)
    for band in range(6):
        assert out[band].ravel().tolist() == pytest.approx([0.0, 1.0], abs=1e-3)


def test_prithvi_token_count_guard_accepts_grid_plus_class_token() -> None:
    # tokens_to_grid drops leading class/register tokens, so an exact grid or a
    # grid + a few leading tokens is fine. A 224 px tile at patch size 16 is a
    # 14x14 = 196 patch grid; the ViT prepends one class token -> 197. Torch-free.
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend

    backend = PrithviBackend()
    backend._check_token_count(196, 14, 14)  # exact patch grid
    backend._check_token_count(197, 14, 14)  # + one class token


def test_prithvi_token_count_guard_rejects_mismatch() -> None:
    # A gross mismatch (patch size disagrees with the checkpoint, or the encoder
    # masked patches) must fail loud instead of silently mis-slicing the grid.
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend

    backend = PrithviBackend()
    with pytest.raises(RuntimeError, match="patch grid expects"):
        backend._check_token_count(784, 14, 14)  # far too many tokens
    with pytest.raises(RuntimeError, match="patch grid expects"):
        backend._check_token_count(100, 14, 14)  # fewer tokens than patches


def test_prithvi_load_model_rejects_missing_local_weights() -> None:
    # An explicit local checkpoint path that does not exist must fail with a clear
    # FileNotFoundError up front (via resolve_local), before TerraTorch is imported
    # — mirroring Clay — rather than failing opaquely deep in the registry build.
    # Torch/terratorch-free: resolve_local runs first and raises, so calling
    # _load_model directly never reaches the (uninstalled) terratorch import.
    from eo_inference_toolbox.eocore.backends.prithvi import PrithviBackend

    backend = PrithviBackend(weights="/definitely/not/a/real/checkpoint.pt")
    with pytest.raises(FileNotFoundError, match="weights not found"):
        backend._load_model()


def test_no_backend_claims_unit_normalized() -> None:
    # The whole L2-normalise-before-write contract depends on this being False.
    for backend_id in BACKEND_IDS:
        assert backend_info(backend_id).is_unit_normalized is False


def test_backend_info_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown backend"):
        backend_info("landsat-gpt")


def test_descriptor_from_info_carries_facts() -> None:
    desc = descriptor_from_info(backend_info("prithvi"))
    assert desc.backend_id == "prithvi"
    assert desc.temporal is True  # Prithvi is the multi-temporal backend
    assert desc.capabilities.needs_auth is True  # HLS default collection needs auth
    assert desc.n_bands == backend_info("prithvi").embedding_dim


def test_descriptor_from_info_band_override() -> None:
    desc = descriptor_from_info(backend_info("clay"), n_bands=64)
    assert desc.n_bands == 64
    assert desc.temporal is False


def test_resolve_bands_default_and_override() -> None:
    assert resolve_bands("clay", None) == backend_info("clay").default_bands
    assert resolve_bands("clay", ["red", "green"]) == ("red", "green")


def test_strict_bands_flag_matches_backend_convention() -> None:
    # Prithvi standardises positionally and Satlas's RGB checkpoint takes exactly
    # three channels, so both are strict; Clay consumes bands by name and is not.
    assert backend_info("prithvi").strict_bands is True
    assert backend_info("satlas").strict_bands is True
    assert backend_info("clay").strict_bands is False


def test_band_requirement_error_prithvi_on_rgb_points_at_satlas() -> None:
    # The reported failure: imagery fetched with Satlas (3 RGB bands) then embedded
    # with Prithvi. Instead of the model's opaque "got 3", name the mismatch and
    # suggest the backend whose defaults match the imagery.
    message = band_requirement_error("prithvi", ("red", "green", "blue"))
    assert message is not None
    assert "3 band(s) (red, green, blue)" in message
    assert "Prithvi-EO-2.0 (IBM + NASA)" in message
    assert "6 band(s) (blue, green, red, narrow_nir, swir1, swir2)" in message
    assert "SatlasPretrain (Allen Institute / AI2)" in message


def test_band_requirement_error_satlas_on_rgbn_points_at_clay() -> None:
    # A 4-band RGBN raster (Clay's default set) fed to the 3-channel Satlas RGB
    # checkpoint: flag it up front and point at Clay.
    message = band_requirement_error("satlas", ("red", "green", "blue", "nir"))
    assert message is not None
    assert "4 band(s) (red, green, blue, nir)" in message
    assert "3 band(s) (red, green, blue)" in message
    assert "Clay v1.5" in message


def test_band_requirement_error_none_when_compatible_or_flexible() -> None:
    from eo_inference_toolbox.eocore.stac import PRITHVI_BANDS

    # Exact match on a strict backend -> no error.
    assert band_requirement_error("prithvi", PRITHVI_BANDS) is None
    assert band_requirement_error("satlas", ("red", "green", "blue")) is None
    # Clay is flexible: any band set is accepted, so never an error here.
    assert band_requirement_error("clay", ("red", "green", "blue")) is None
    assert band_requirement_error("clay", ("swir1", "swir2")) is None


def test_band_requirement_error_wrong_order_has_no_false_suggestion() -> None:
    # Six bands, right names, wrong order: still a real mismatch for Prithvi (it
    # standardises positionally), but no backend's defaults are exactly this set,
    # so the message falls back to the band-order override rather than a bad match.
    scrambled = ("red", "green", "blue", "narrow_nir", "swir1", "swir2")
    message = band_requirement_error("prithvi", scrambled)
    assert message is not None
    assert "Set the band-order override to 'blue, green, red, narrow_nir, swir1, swir2'" in message
    assert "match the" not in message


def test_load_backend_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown backend"):
        load_backend("not-a-backend")


def test_default_input_scale_is_positive_and_registered() -> None:
    # Every backend documents the multiplier it applies to raw pixels (H-7).
    assert backend_info("clay").default_input_scale == pytest.approx(1.0 / 10000.0)
    assert backend_info("prithvi").default_input_scale == pytest.approx(1.0e-4)
    assert backend_info("satlas").default_input_scale == pytest.approx(1.0 / 255.0)


def test_scale_sanity_warning_plausible_returns_none() -> None:
    # Sentinel-2 L2A DN (max ~4000) at the Clay default (1/10000) lands near 0.4.
    assert scale_sanity_warning(4000.0, 1.0 / 10000.0) is None


def test_scale_sanity_warning_flags_already_reflectance() -> None:
    # Reflectance imagery (max ~0.4) fed at the DN default scales to ~4e-5 -> warn.
    message = scale_sanity_warning(0.4, 1.0 / 10000.0)
    assert message is not None
    assert "Input scale" in message


def test_scale_sanity_warning_flags_eight_bit_on_dn_backend() -> None:
    # 8-bit imagery (max 255) fed to a DN backend scales to ~0.025 -> warn.
    assert scale_sanity_warning(255.0, 1.0 / 10000.0) is not None


def test_scale_sanity_warning_flags_far_too_large() -> None:
    # Raw DN with no scaling (scale 1.0) scales to thousands -> warn.
    message = scale_sanity_warning(8000.0, 1.0)
    assert message is not None
    assert "above" in message


def test_scale_sanity_warning_ignores_degenerate_inputs() -> None:
    assert scale_sanity_warning(0.0, 1.0 / 10000.0) is None
    assert scale_sanity_warning(float("nan"), 1.0 / 10000.0) is None
    assert scale_sanity_warning(4000.0, 0.0) is None


# -- E-2 batched inference dispatch -------------------------------------------
_META = TileMeta(bands=("red",), gsd=10.0)


class _LoopBackend:
    """A per-tile-only fake backend (no ``embed_tiles``) — sums bands to 1 band."""

    backend_id = "loop"
    display_name = "Loop"

    def descriptor(self) -> Any:  # not exercised here
        raise NotImplementedError

    def required_bands(self) -> tuple[str, ...]:
        return ("red",)

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        out: NDArray[Any] = np.asarray(image, dtype=np.float32).sum(axis=0, keepdims=True)
        return out


class _BatchBackend(_LoopBackend):
    """A batch-capable fake that records which path (batch vs per-tile) ran."""

    backend_id = "batch"

    def __init__(self) -> None:
        self.batch_calls = 0
        self.tile_calls = 0

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        self.tile_calls += 1
        return super().embed_tile(image, meta)

    def embed_tiles(self, images: Sequence[NDArray[Any]], meta: TileMeta) -> list[NDArray[Any]]:
        self.batch_calls += 1
        return [np.asarray(im, dtype=np.float32).sum(axis=0, keepdims=True) for im in images]


def test_embed_tile_batch_single_item_uses_per_tile_path() -> None:
    backend = _BatchBackend()
    chips = [np.ones((2, 3, 3), dtype=np.float32)]
    out = embed_tile_batch(backend, chips, _META)
    assert len(out) == 1
    # A one-tile batch is always the exact v0.1.0 per-tile route.
    assert backend.tile_calls == 1
    assert backend.batch_calls == 0


def test_embed_tile_batch_uses_batch_path_for_multiple() -> None:
    backend = _BatchBackend()
    chips = [np.ones((2, 3, 3), dtype=np.float32) for _ in range(3)]
    out = embed_tile_batch(backend, chips, _META)
    assert len(out) == 3
    assert backend.batch_calls == 1
    assert backend.tile_calls == 0


def test_embed_tile_batch_falls_back_to_loop_without_batch_method() -> None:
    backend = _LoopBackend()
    chips = [np.arange(8, dtype=np.float32).reshape(2, 2, 2) + k for k in range(2)]
    out = embed_tile_batch(backend, chips, _META)
    assert len(out) == 2
    for chip, embedded in zip(chips, out, strict=True):
        assert np.allclose(embedded, chip.sum(axis=0, keepdims=True))


def test_embed_tile_batch_rejects_bad_batch_length() -> None:
    class _ShortBatch(_LoopBackend):
        def embed_tiles(self, images: Sequence[NDArray[Any]], meta: TileMeta) -> list[NDArray[Any]]:
            return [np.zeros((1, 3, 3), dtype=np.float32)]  # too few outputs

    chips = [np.ones((2, 3, 3), dtype=np.float32) for _ in range(2)]
    with pytest.raises(RuntimeError, match="embeddings for a batch"):
        embed_tile_batch(_ShortBatch(), chips, _META)


def test_batch_capability_isinstance_checks() -> None:
    assert isinstance(_BatchBackend(), BatchEmbeddingBackend)
    assert not isinstance(_LoopBackend(), BatchEmbeddingBackend)
    # A batch-capable backend is still a plain EmbeddingBackend.
    assert isinstance(_BatchBackend(), EmbeddingBackend)
    assert isinstance(_LoopBackend(), EmbeddingBackend)
