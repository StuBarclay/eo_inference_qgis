"""Byte-identity parity tests for the low-memory streamed analytics.

The opt-in low-memory path processes an embedding raster one tile at a time
instead of reading the whole ``(D, H, W)`` cube into RAM. For an analytic whose
output is a **per-pixel function of a fixed transform**, tiling must not change a
single output byte: this module pins that guarantee for the pure-core transforms
behind the four per-pixel analytics (:mod:`eo_inference_toolbox.eocore`) by
evaluating them whole-cube versus a faithful mirror of the streamed tiling loop and
asserting exact equality. The QGIS-facing algorithm layer that wires these transforms
to Processing is not importable in the sandbox (it imports ``qgis``), so its
end-to-end behaviour is left to the manual [SMOKE_TEST.md](../docs/SMOKE_TEST.md).
Everything here is torch-, GDAL- and QGIS-free.

Fit-vs-apply asymmetry (why some analytics are only tested at the apply stage)
------------------------------------------------------------------------------
Two of the streamed analytics fit a model before applying it, and only the
*apply* stage is byte-identical:

* **Cluster** — the streamed fit is ``MiniBatchKMeans.partial_fit`` over *every*
  pixel, whereas the whole-cube fit is ``KMeans`` on a subsample, so the two
  fitted models (and hence outputs) legitimately differ. Given a *fixed*
  estimator, though, the per-pixel *prediction* is byte-identical when streamed,
  which is what these tests cover (with a deterministic fake predictor).
* **Embedding→RGB** — the streamed preview fits its PCA / percentile-clip bounds
  on a pixel *sample*; the whole-cube preview fits on every pixel. Once a
  :class:`RgbTransform` is fixed, applying it tile-by-tile is byte-identical to
  applying it whole, which is what is asserted here.

Similarity, change and classify have no such gap: their streamed output is
byte-identical to the whole-cube output end to end (classify's estimator is fixed
before the scene pass), so those are exercised directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from conftest import make_cube, unit_cube
from numpy.typing import NDArray

from eo_inference_toolbox.eocore.change import distance_map
from eo_inference_toolbox.eocore.ml import (
    cube_to_matrix,
    predict_in_chunks,
    scatter_to_grid,
    valid_mask_from_nodata,
)
from eo_inference_toolbox.eocore.rgb import (
    apply_rgb_transform,
    embedding_to_rgb,
    fit_rgb_transform,
)
from eo_inference_toolbox.eocore.similarity import cosine_similarity_map, reference_from_points
from eo_inference_toolbox.eocore.streaming import map_windows, map_windows_multi, plan_windows
from eo_inference_toolbox.eocore.tiling import Window

#: A nodata sentinel; matches the algorithms' DEFAULT_NODATA style but any value works.
NODATA = -9999.0
#: The classify / cluster algorithms fill invalid pixels with -1 (their CLASS_NODATA /
#: CLUSTER_NODATA constants, which live in the QGIS-bound algorithm modules); hardcoded
#: here so the pure-core parity test needs no qgis import.
LABEL_FILL = -1


def _reader(cube: NDArray[Any]) -> Callable[[Window], NDArray[Any]]:
    """A fake windowed reader over an in-memory ``(bands, H, W)`` cube (like the real one)."""

    def read(window: Window) -> NDArray[Any]:
        return cube[
            :,
            window.row_off : window.row_end,
            window.col_off : window.col_end,
        ]

    return read


def _cube_with_nodata(n_bands: int, height: int, width: int, *, seed: int = 0) -> NDArray[Any]:
    """A float32 cube seeded with an all-nodata block and one non-finite pixel.

    Exercising the validity mask matters: the parity guarantee must hold *including*
    how invalid pixels are treated, and the mask must agree pixel-for-pixel whether
    computed whole or per tile.
    """
    cube = make_cube(n_bands, height, width, seed=seed)
    cube[:, :3, :5] = NODATA  # a block of all-nodata pixels -> invalid
    cube[0, height - 2, width - 3] = np.nan  # a single non-finite pixel -> invalid
    return cube


class _PerRowLabeler:
    """A deterministic, per-row classifier/clusterer stand-in (no sklearn needed).

    ``predict`` maps each sample row to an int label using only that row, so it is a
    pure per-pixel function — exactly the property that makes the streamed apply
    stage byte-identical to the whole-cube one. Stands in for a fitted
    RandomForest / KMeans whose ``predict`` is likewise per-row.
    """

    def predict(self, matrix: NDArray[Any]) -> NDArray[Any]:
        row_sum = np.asarray(matrix, dtype=np.float64).sum(axis=1)
        labels: NDArray[Any] = np.clip(np.floor(row_sum).astype(np.int64) + 3, 0, 6)
        return labels


def _label_whole(cube: NDArray[Any], estimator: _PerRowLabeler, fill: int) -> NDArray[Any]:
    """Whole-cube label map — the exact pure-core path classify/cluster run by default."""
    mask = valid_mask_from_nodata(cube, NODATA)
    valid = cube_to_matrix(cube)[mask.reshape(-1)]
    preds = predict_in_chunks(valid, estimator.predict)
    return scatter_to_grid(preds, mask, fill=fill, dtype=np.int32)


def _label_block(estimator: _PerRowLabeler, fill: int) -> Callable[[NDArray[Any]], NDArray[Any]]:
    """Per-tile label block — the exact pure-core path classify/cluster stream."""

    def block(tile: NDArray[Any]) -> NDArray[Any]:
        flat = valid_mask_from_nodata(tile, NODATA).reshape(-1)
        out = np.full(flat.shape[0], fill, dtype=np.int32)
        if flat.any():
            preds = predict_in_chunks(cube_to_matrix(tile)[flat], estimator.predict)
            out[flat] = preds.astype(np.int32)
        return out.reshape(1, tile.shape[1], tile.shape[2])

    return block


# --- similarity (template: per-pixel map, byte-identical end to end) --------


def test_similarity_stream_matches_whole_cube_template() -> None:
    # The template the other apply-stage tests mirror: a per-pixel map streamed
    # tile-by-tile must reproduce the whole-cube result exactly. Covers both the
    # assume-unit fast path and the per-pixel L2-normalise path (independent per
    # pixel, so still tile-invariant).
    unit = unit_cube(8, 40, 52, seed=1)
    raw = make_cube(8, 40, 52, seed=2)
    for cube, assume_unit in ((unit, True), (raw, False)):
        ref = reference_from_points(cube, [(20, 25), (5, 6)], normalize=not assume_unit)

        def block(
            tile: NDArray[Any], _r: NDArray[Any] = ref, _a: bool = assume_unit
        ) -> NDArray[Any]:
            return cosine_similarity_map(tile, _r, assume_unit=_a)[None, :, :]

        whole = cosine_similarity_map(cube, ref, assume_unit=assume_unit)[None, :, :]
        for tile in (16, 24, 999):
            streamed = map_windows(
                plan_windows(52, 40, tile),
                _reader(cube),
                block,
                out_bands=1,
                height=40,
                width=52,
                dtype=np.float32,
            )
            assert streamed is not None
            np.testing.assert_array_equal(streamed, whole.astype(np.float32))


# --- change (multi-input per-pixel map, byte-identical end to end) ----------


def test_change_stream_matches_whole_cube_template() -> None:
    # Change reads N aligned cubes; the multi-input streamer must match the
    # whole-scene per-pixel distance exactly, for both metrics.
    a = make_cube(6, 36, 44, seed=3)
    b = make_cube(6, 36, 44, seed=4)
    for metric in ("cosine", "euclidean"):

        def block(tiles: list[NDArray[Any]], _m: str = metric) -> NDArray[Any]:
            return distance_map(tiles[0], tiles[1], metric=_m)[None, :, :]

        whole = distance_map(a, b, metric=metric)[None, :, :]
        for tile in (12, 16, 999):
            streamed = map_windows_multi(
                plan_windows(44, 36, tile),
                [_reader(a), _reader(b)],
                block,
                out_bands=1,
                height=36,
                width=44,
                dtype=np.float32,
            )
            assert streamed is not None
            np.testing.assert_array_equal(streamed, whole.astype(np.float32))


# --- classify (apply stage: byte-identical; estimator fixed before the pass) -


def test_classify_apply_stream_matches_whole_cube() -> None:
    cube = _cube_with_nodata(10, 40, 50, seed=5)
    estimator = _PerRowLabeler()
    whole = _label_whole(cube, estimator, LABEL_FILL)
    streamed = map_windows(
        plan_windows(50, 40, 16),
        _reader(cube),
        _label_block(estimator, LABEL_FILL),
        out_bands=1,
        height=40,
        width=50,
        dtype=np.int32,
        fill=LABEL_FILL,
    )
    assert streamed is not None
    # Byte-identical class map, invalid pixels included.
    np.testing.assert_array_equal(streamed[0], whole)
    assert streamed.dtype == np.int32
    # The all-nodata block is labelled with the fill in both paths.
    assert (streamed[0][:3, :5] == LABEL_FILL).all()
    assert (whole[:3, :5] == LABEL_FILL).all()


def test_classify_apply_stream_is_tile_size_invariant() -> None:
    cube = _cube_with_nodata(12, 47, 53, seed=6)
    estimator = _PerRowLabeler()
    whole = _label_whole(cube, estimator, LABEL_FILL)
    for tile in (8, 16, 32, 999):
        streamed = map_windows(
            plan_windows(53, 47, tile),
            _reader(cube),
            _label_block(estimator, LABEL_FILL),
            out_bands=1,
            height=47,
            width=53,
            dtype=np.int32,
            fill=LABEL_FILL,
        )
        assert streamed is not None
        np.testing.assert_array_equal(streamed[0], whole)


# --- cluster (apply stage byte-identical; see the module docstring on the fit) -


def test_cluster_apply_stream_matches_whole_cube() -> None:
    # NOTE: only the *apply* stage is byte-identical. Cluster's streamed fit
    # (MiniBatch over all pixels) differs from the whole-cube fit (KMeans on a
    # subsample); see the module docstring. With the estimator fixed, prediction
    # is a per-pixel function, so streamed == whole to the byte. Cluster runs the
    # identical pure-core apply block as classify (only the nodata fill name
    # differs, and both are -1), so this pins the cluster apply path too.
    cube = _cube_with_nodata(9, 38, 41, seed=7)
    estimator = _PerRowLabeler()
    whole = _label_whole(cube, estimator, LABEL_FILL)
    for tile in (16, 999):
        streamed = map_windows(
            plan_windows(41, 38, tile),
            _reader(cube),
            _label_block(estimator, LABEL_FILL),
            out_bands=1,
            height=38,
            width=41,
            dtype=np.int32,
            fill=LABEL_FILL,
        )
        assert streamed is not None
        np.testing.assert_array_equal(streamed[0], whole)


# --- embedding -> RGB (apply stage byte-identical; see docstring on the fit) --


def _fit_transform_on_all_valid(cube: NDArray[Any], method: str, bands: Any = None) -> Any:
    """Fit an RgbTransform on every valid pixel (what the whole-cube preview does)."""
    mask = valid_mask_from_nodata(cube, NODATA)
    valid = cube_to_matrix(cube)[mask.reshape(-1)]
    return fit_rgb_transform(valid, method=method, bands=bands, n_bands=cube.shape[0])


def test_rgb_apply_stream_matches_whole_cube_pca() -> None:
    cube = _cube_with_nodata(12, 40, 50, seed=8)
    transform = _fit_transform_on_all_valid(cube, "pca")
    whole = apply_rgb_transform(cube, transform, nodata=NODATA)

    def block(tile: NDArray[Any]) -> NDArray[Any]:
        return apply_rgb_transform(tile, transform, nodata=NODATA)

    for tile in (16, 24, 999):
        streamed = map_windows(
            plan_windows(50, 40, tile),
            _reader(cube),
            block,
            out_bands=3,
            height=40,
            width=50,
            dtype=np.uint8,
            fill=0,
        )
        assert streamed is not None
        np.testing.assert_array_equal(streamed, whole)
    # The whole-cube convenience wrapper fits on the same valid pixels, so it
    # produces the identical transform and hence the identical image.
    np.testing.assert_array_equal(whole, embedding_to_rgb(cube, method="pca", nodata=NODATA))


def test_rgb_apply_stream_matches_whole_cube_bands() -> None:
    cube = _cube_with_nodata(12, 36, 44, seed=9)
    transform = _fit_transform_on_all_valid(cube, "bands", bands=(0, 5, 11))
    whole = apply_rgb_transform(cube, transform, nodata=NODATA)

    def block(tile: NDArray[Any]) -> NDArray[Any]:
        return apply_rgb_transform(tile, transform, nodata=NODATA)

    for tile in (12, 16, 999):
        streamed = map_windows(
            plan_windows(44, 36, tile),
            _reader(cube),
            block,
            out_bands=3,
            height=36,
            width=44,
            dtype=np.uint8,
            fill=0,
        )
        assert streamed is not None
        np.testing.assert_array_equal(streamed, whole)
