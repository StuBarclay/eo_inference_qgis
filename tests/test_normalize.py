"""Tests for L2 normalisation — the correctness linchpin of the toolbox."""

from __future__ import annotations

import numpy as np
from conftest import make_cube

from eo_inference_toolbox.eocore.descriptor import EmbeddingDescriptor
from eo_inference_toolbox.eocore.normalize import (
    ensure_unit,
    l2_normalize,
    normalize_rows,
)


def test_l2_normalize_gives_unit_length() -> None:
    cube = make_cube(6, 8, 8, seed=1)
    out = l2_normalize(cube)
    norms = np.sqrt(np.sum(out * out, axis=0))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_leaves_zero_pixels_at_zero() -> None:
    cube = np.zeros((4, 3, 3), dtype=np.float32)
    out = l2_normalize(cube)
    assert np.all(out == 0.0)


def test_l2_normalize_preserves_float32() -> None:
    cube = make_cube(5, 4, 4).astype(np.float32)
    assert l2_normalize(cube).dtype == np.float32


def test_normalize_rows_unit_length() -> None:
    matrix = np.arange(1, 13, dtype=np.float64).reshape(4, 3)
    out = normalize_rows(matrix)
    assert np.allclose(np.sqrt(np.sum(out * out, axis=1)), 1.0)


def test_ensure_unit_normalizes_when_flag_false() -> None:
    cube = make_cube(3, 5, 5, seed=2)
    desc = EmbeddingDescriptor(backend_id="clay", n_bands=3, is_unit_normalized=False)
    out = ensure_unit(cube, desc)
    norms = np.sqrt(np.sum(out * out, axis=0))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_ensure_unit_passthrough_when_flag_true() -> None:
    cube = make_cube(3, 5, 5, seed=3) * 7.0
    desc = EmbeddingDescriptor(backend_id="clay", n_bands=3, is_unit_normalized=True)
    out = ensure_unit(cube, desc)
    # Not re-normalised: still carries the original magnitude.
    assert np.allclose(out, cube.astype(out.dtype))
