"""End-to-end embed orchestration with a deterministic dummy backend (H-12 / E-13).

The real ``embed`` algorithm (``algorithms/embed.py``) composes a fixed pipeline:
window the scene, embed each tile through the backend, mosaic with overlap
averaging, L2-normalise, stamp nodata into uncovered pixels, and replace the
descriptor's band count / unit flag. That algorithm imports ``qgis.core``, so the
full path was previously only exercised by the manual smoke test (the real
backends need torch + weights).

Here a torch-free dummy :class:`EmbeddingBackend` lets the *same* composition run
end-to-end in the bare sandbox. We drive the eocore call sequence the algorithm
uses (``iter_windows`` -> ``run_windowed_embed`` -> ``embed_tile_batch`` ->
``MosaicAccumulator``) and then hand the mosaic to the very same
:func:`finalize_embedding` the algorithm calls (normalise -> nodata -> descriptor
replace), so the finalize step is exercised here rather than reimplemented (M7).
We assert the invariants that matter: unit-length output, correct band count, a
stamped descriptor, nodata in uncovered pixels, and byte-identical results
whether tiles are embedded one-at-a-time or in a batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eo_inference_toolbox.eocore.backends import (
    BatchEmbeddingBackend,
    EmbeddingBackend,
    TileMeta,
    embed_tile_batch,
)
from eo_inference_toolbox.eocore.descriptor import DEFAULT_NODATA, EmbeddingDescriptor
from eo_inference_toolbox.eocore.runner import finalize_embedding, run_windowed_embed
from eo_inference_toolbox.eocore.tiling import MosaicAccumulator, Window, iter_windows

META = TileMeta(bands=("red", "green", "blue"), gsd=10.0, lon=150.0, lat=-33.0)


class DummyBackend:
    """A deterministic, torch-free :class:`EmbeddingBackend` for orchestration tests.

    Each ``(C, h, w)`` tile maps to a ``(D, h, w)`` cube whose band ``d`` equals
    the per-pixel channel mean plus ``d``. The result is content-dependent (so
    adjacent tiles differ and overlaps genuinely average) and deliberately
    *non*-unit-length (so L2-normalisation has visible work to do). ``embed_tiles``
    computes the identical result over a stacked batch, so the per-tile and batch
    code paths must agree — the E-2 correctness guarantee, end to end.
    """

    backend_id = "dummy"
    display_name = "Dummy (test-only)"

    def __init__(self, *, embedding_dim: int = 5, **_ignored: Any) -> None:
        self._embedding_dim = int(embedding_dim)

    def descriptor(self) -> EmbeddingDescriptor:
        return EmbeddingDescriptor(
            backend_id=self.backend_id,
            n_bands=self._embedding_dim,
            is_unit_normalized=False,
            resolution_m=10.0,
        )

    def required_bands(self) -> tuple[str, ...]:
        return ("red", "green", "blue")

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected (C, H, W) tile, got shape {arr.shape}")
        base = arr.mean(axis=0)  # (h, w)
        offsets = np.arange(self._embedding_dim, dtype=np.float32)
        out: NDArray[Any] = base[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
        return out.astype(np.float32)

    def embed_tiles(
        self, images: list[NDArray[Any]] | tuple[NDArray[Any], ...], meta: TileMeta
    ) -> list[NDArray[Any]]:
        arrs = [np.asarray(image, dtype=np.float32) for image in images]
        if len({arr.shape for arr in arrs}) != 1:
            return [self.embed_tile(arr, meta) for arr in arrs]
        batch = np.stack(arrs, axis=0)  # (B, C, h, w)
        base = batch.mean(axis=1)  # (B, h, w)
        offsets = np.arange(self._embedding_dim, dtype=np.float32)
        cubes = base[:, np.newaxis, :, :] + offsets[np.newaxis, :, np.newaxis, np.newaxis]
        return [cube.astype(np.float32) for cube in cubes]


def _read_window_factory(n_in: int) -> Any:
    """A GDAL-free reader: fabricate content that varies with window position.

    Distinct per-window values make overlap averaging and per-tile/batch parity
    non-trivial to satisfy (constant tiles would hide a broken mosaic).
    """

    def read_window(window: Window) -> NDArray[Any]:
        value = 1.0 + window.col_off + 10.0 * window.row_off
        return np.full((n_in, window.height, window.width), value, dtype=np.float32)

    return read_window


def _embed_pipeline(
    backend: EmbeddingBackend,
    *,
    width: int,
    height: int,
    tile: int,
    overlap: int,
    batch_size: int,
    n_in: int = 3,
    windows: list[Window] | None = None,
) -> tuple[NDArray[Any], EmbeddingDescriptor, NDArray[Any]]:
    """Reproduce ``EmbedAlgorithm``'s eocore composition, minus GDAL/QGIS.

    The tile -> embed -> mosaic half is run here; the normalise -> nodata -> stamp
    tail is delegated to the *same* :func:`finalize_embedding` the algorithm calls
    (M7), so this test exercises the real finalize code rather than a copy that
    could silently drift from it.
    """
    if windows is None:
        windows = list(iter_windows(width, height, tile, overlap))
    read_window = _read_window_factory(n_in)

    def embed_batch(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
        tiles = [np.asarray(chip, dtype=np.float32) for chip in chips]
        return embed_tile_batch(backend, tiles, META)

    mosaic = run_windowed_embed(
        windows, height, width, read_window, embed_batch, batch_size=batch_size
    )
    assert mosaic is not None
    normalized, coverage, descriptor = finalize_embedding(mosaic, backend.descriptor())
    return normalized, descriptor, coverage


def test_dummy_backend_satisfies_the_protocols() -> None:
    backend = DummyBackend()
    assert isinstance(backend, EmbeddingBackend)
    assert isinstance(backend, BatchEmbeddingBackend)  # it has embed_tiles


def test_full_pipeline_produces_unit_normalised_cube() -> None:
    backend = DummyBackend(embedding_dim=5)
    normalized, descriptor, coverage = _embed_pipeline(
        backend, width=48, height=32, tile=16, overlap=4, batch_size=1
    )

    # Shape and band count come from the backend's embedding dimension.
    assert normalized.shape == (5, 32, 48)
    assert descriptor.n_bands == 5
    assert descriptor.backend_id == "dummy"
    assert descriptor.is_unit_normalized is True  # stamped after normalisation

    # iter_windows covers the whole scene, so every pixel is written ...
    assert coverage.all()
    assert not np.any(normalized == DEFAULT_NODATA)
    # ... and every pixel vector is unit length (the downstream cosine == dot rule).
    norms = np.sqrt((normalized**2).sum(axis=0))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_batch_path_matches_per_tile_path() -> None:
    # Same scene, same backend, different batch sizes: the E-2 invariant is that
    # embed_tile_batch's stacked path returns exactly the per-tile results.
    per_tile, _, _ = _embed_pipeline(
        DummyBackend(), width=48, height=32, tile=16, overlap=4, batch_size=1
    )
    batched, _, _ = _embed_pipeline(
        DummyBackend(), width=48, height=32, tile=16, overlap=4, batch_size=4
    )
    assert np.array_equal(per_tile, batched)


def test_overlap_region_averages_adjacent_tiles() -> None:
    # Two horizontally-overlapping windows carrying different fabricated values;
    # the mosaic must average them in the overlap before normalisation. We check
    # the pre-normalisation mosaic directly via a MosaicAccumulator round-trip.
    backend = DummyBackend(embedding_dim=1)  # 1 band => value == channel mean + 0
    windows = [Window(0, 0, 8, 4), Window(4, 0, 8, 4)]  # overlap columns 4..7
    read_window = _read_window_factory(3)

    def embed_batch(chips: list[NDArray[Any]]) -> list[NDArray[Any]]:
        return embed_tile_batch(backend, chips, META)

    mosaic = run_windowed_embed(windows, 4, 12, read_window, embed_batch, batch_size=1)
    assert mosaic is not None
    row = mosaic.result()[0, 0]
    left_value = 1.0  # window col_off=0 -> fabricated 1.0
    right_value = 5.0  # window col_off=4 -> fabricated 5.0
    assert row[0] == pytest.approx(left_value)
    assert row[-1] == pytest.approx(right_value)
    # Overlap columns saw both tiles -> plain average.
    assert np.allclose(row[4:8], (left_value + right_value) / 2.0)


def test_uncovered_pixels_get_nodata() -> None:
    # Hand-built windows that leave a gap on the right of the scene, so the
    # orchestration's nodata-stamping branch (normalized[:, ~coverage] = NODATA)
    # is exercised — iter_windows itself never leaves gaps.
    backend = DummyBackend(embedding_dim=3)
    width, height = 20, 4
    windows = [Window(0, 0, 8, 4)]  # covers only columns 0..7
    normalized, _, coverage = _embed_pipeline(
        backend,
        width=width,
        height=height,
        tile=8,
        overlap=0,
        batch_size=1,
        windows=windows,
    )
    assert coverage[:, :8].all()
    assert not coverage[:, 8:].any()
    # Uncovered pixels carry the nodata sentinel across every band ...
    assert np.all(normalized[:, :, 8:] == DEFAULT_NODATA)
    # ... while covered pixels are unit-normalised, not nodata.
    covered_norms = np.sqrt((normalized[:, :, :8] ** 2).sum(axis=0))
    assert np.allclose(covered_norms, 1.0, atol=1e-5)


def test_pipeline_is_deterministic() -> None:
    first, _, _ = _embed_pipeline(
        DummyBackend(), width=32, height=16, tile=16, overlap=4, batch_size=2
    )
    second, _, _ = _embed_pipeline(
        DummyBackend(), width=32, height=16, tile=16, overlap=4, batch_size=2
    )
    assert np.array_equal(first, second)


def test_finalize_embedding_normalises_stamps_and_flags() -> None:
    # Exercise finalize_embedding in isolation on a hand-built mosaic: a 2-band,
    # 1x3 scene whose first two columns are covered by one window (with deliberately
    # non-unit vectors) and whose last column no window ever touched.
    mosaic = MosaicAccumulator(n_bands=2, height=1, width=3)
    # Band 0 = [3, 0], band 1 = [4, 5] over the 1x2 covered window.
    mosaic.add(Window(0, 0, 2, 1), np.array([[[3.0, 0.0]], [[4.0, 5.0]]], dtype=np.float32))
    # A placeholder descriptor whose n_bands is wrong on purpose (99), to prove
    # finalize re-derives the band count from the real output depth.
    descriptor = EmbeddingDescriptor(
        backend_id="dummy", n_bands=99, is_unit_normalized=False, resolution_m=10.0
    )

    normalized, coverage, final = finalize_embedding(mosaic, descriptor)

    # Descriptor is stamped from the true depth and flagged unit-normalised.
    assert final.n_bands == 2
    assert final.is_unit_normalized is True
    assert final.backend_id == "dummy"
    # Covered pixels are unit length ([3,4]/5 and [0,5]/5); uncovered = nodata.
    assert coverage.tolist() == [[True, True, False]]
    np.testing.assert_allclose(normalized[:, 0, 0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(normalized[:, 0, 1], [0.0, 1.0], atol=1e-6)
    assert np.all(normalized[:, 0, 2] == DEFAULT_NODATA)


def test_finalize_embedding_stamps_nonfinite_as_nodata() -> None:
    # A backend that emits a non-finite value for one covered pixel must not leak
    # NaN/Inf into the raster: finalize folds finiteness into coverage so the bad
    # pixel is written as nodata and dropped from coverage, even though a tile did
    # "cover" it (weight > 0, so the uncovered-stamp alone would miss it).
    mosaic = MosaicAccumulator(n_bands=2, height=1, width=3)
    # Cover all three columns; the middle pixel carries NaN in both bands.
    tile = np.array([[[3.0, np.nan, 3.0]], [[4.0, np.nan, 4.0]]], dtype=np.float32)
    mosaic.add(Window(0, 0, 3, 1), tile)
    descriptor = EmbeddingDescriptor(
        backend_id="dummy", n_bands=2, is_unit_normalized=False, resolution_m=10.0
    )

    normalized, coverage, _final = finalize_embedding(mosaic, descriptor)

    # The non-finite pixel is excluded from coverage and stamped nodata ...
    assert coverage.tolist() == [[True, False, True]]
    assert np.all(normalized[:, 0, 1] == DEFAULT_NODATA)
    # ... while the finite covered pixels are unit-normalised ([3,4]/5).
    np.testing.assert_allclose(normalized[:, 0, 0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(normalized[:, 0, 2], [0.6, 0.8], atol=1e-6)
