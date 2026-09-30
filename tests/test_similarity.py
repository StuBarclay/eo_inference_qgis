"""Tests for the per-pixel cosine similarity map."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cube, unit_cube

from eo_inference_toolbox.eocore.normalize import l2_normalize
from eo_inference_toolbox.eocore.similarity import (
    cosine_similarity_map,
    reference_from_points,
)


def test_identical_reference_scores_one() -> None:
    cube = unit_cube(8, 4, 4, seed=5)
    ref = cube[:, 0, 0]
    sim = cosine_similarity_map(cube, ref, assume_unit=True)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(sim <= 1.0 + 1e-5)
    assert np.all(sim >= -1.0 - 1e-5)


def test_assume_unit_false_normalizes_internally() -> None:
    cube = make_cube(6, 5, 5, seed=6) * 3.0
    ref = cube[:, 1, 1]
    sim = cosine_similarity_map(cube, ref, assume_unit=False)
    # The pixel equal to the (scaled) reference is perfectly similar.
    assert sim[1, 1] == pytest.approx(1.0, abs=1e-5)


def test_matches_manual_normalized_dot() -> None:
    cube = make_cube(4, 3, 3, seed=7)
    ref = cube[:, 2, 2]
    got = cosine_similarity_map(cube, ref, assume_unit=False)
    norm_cube = l2_normalize(cube)
    norm_ref = ref / np.sqrt(np.sum(ref * ref))
    expected = np.tensordot(norm_ref, norm_cube, axes=([0], [0]))
    assert np.allclose(got, expected, atol=1e-5)


def test_reference_length_mismatch_raises() -> None:
    cube = make_cube(4, 3, 3)
    with pytest.raises(ValueError, match="reference length"):
        cosine_similarity_map(cube, np.ones(3, dtype=np.float32))


def test_reference_from_points_is_unit_by_default() -> None:
    cube = make_cube(5, 6, 6, seed=8)
    ref = reference_from_points(cube, [(0, 0), (1, 1), (2, 2)])
    assert np.sqrt(np.sum(ref * ref)) == pytest.approx(1.0, abs=1e-5)


def test_reference_from_points_requires_points() -> None:
    cube = make_cube(5, 6, 6)
    with pytest.raises(ValueError, match="at least one"):
        reference_from_points(cube, [])


def test_reference_from_points_drops_nodata_points() -> None:
    # A reference point landing on the -9999 nodata collar must be dropped, not
    # averaged into the mean (which would silently corrupt the whole map).
    cube = make_cube(4, 3, 3, seed=41)
    cube[:, 0, 0] = -9999.0
    ref = reference_from_points(cube, [(0, 0), (1, 1), (2, 2)], normalize=False, nodata=-9999.0)
    expected = np.mean(np.stack([cube[:, 1, 1], cube[:, 2, 2]], axis=0), axis=0)
    assert np.allclose(ref, expected, atol=1e-6)


def test_reference_from_points_drops_nonfinite_points() -> None:
    # Non-finite pixels are dropped even without a nodata sentinel given.
    cube = make_cube(4, 3, 3, seed=42)
    cube[:, 0, 0] = np.nan
    ref = reference_from_points(cube, [(0, 0), (1, 1)], normalize=False)
    assert np.allclose(ref, cube[:, 1, 1], atol=1e-6)


def test_reference_from_points_all_invalid_raises() -> None:
    cube = make_cube(4, 3, 3, seed=43)
    cube[:, 0, 0] = -9999.0
    cube[:, 1, 1] = -9999.0
    with pytest.raises(ValueError, match="no valid reference"):
        reference_from_points(cube, [(0, 0), (1, 1)], nodata=-9999.0)


def test_cosine_similarity_clamped_to_unit_range() -> None:
    # assume_unit=True skips the internal normalisation, so a deliberately
    # non-unit reference drives the raw dot product outside [-1, 1]; the clamp
    # must contain it so the documented range is a hard guarantee.
    cube = np.full((3, 1, 1), 2.0, dtype=np.float32)  # pixel vector [2, 2, 2]
    over = cosine_similarity_map(cube, np.full(3, 2.0, dtype=np.float32), assume_unit=True)
    assert over[0, 0] == pytest.approx(1.0)  # raw dot 12 -> clamped to 1
    under = cosine_similarity_map(cube, np.full(3, -2.0, dtype=np.float32), assume_unit=True)
    assert under[0, 0] == pytest.approx(-1.0)  # raw dot -12 -> clamped to -1
    assert over.max() <= 1.0
    assert under.min() >= -1.0
