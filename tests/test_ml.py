"""Tests for the sklearn-free ML plumbing (reshape / mask / scatter / encode)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import make_cube
from numpy.typing import NDArray

from eo_inference_toolbox.eocore.ml import (
    classification_metrics,
    confusion_matrix,
    cube_to_matrix,
    encode_labels,
    format_accuracy_report,
    predict_in_chunks,
    scatter_to_grid,
    select_best_index,
    train_val_split,
    valid_mask_from_nodata,
)


def test_cube_to_matrix_roundtrips_shape() -> None:
    cube = make_cube(5, 4, 3, seed=40)
    matrix = cube_to_matrix(cube)
    assert matrix.shape == (12, 5)
    # First matrix row is the (band-wise) first pixel of the cube.
    assert np.allclose(matrix[0], cube[:, 0, 0])


def test_cube_to_matrix_rejects_2d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        cube_to_matrix(np.zeros((4, 4), dtype=np.float32))


def test_valid_mask_flags_nodata_and_naninf() -> None:
    cube = np.ones((2, 2, 2), dtype=np.float32)
    cube[:, 0, 0] = -9999.0
    cube[0, 0, 1] = np.nan
    mask = valid_mask_from_nodata(cube, nodata=-9999.0)
    assert mask[1, 0] and mask[1, 1]
    assert not mask[0, 0]  # all-nodata pixel
    assert not mask[0, 1]  # contains a NaN


def test_encode_labels_preserves_first_seen_order() -> None:
    codes, classes = encode_labels(["water", "urban", "water", "veg", "urban"])
    assert classes == ["water", "urban", "veg"]
    assert list(codes) == [0, 1, 0, 2, 1]


def test_predict_in_chunks_matches_whole() -> None:
    matrix = make_cube(3, 10, 10, seed=41).reshape(3, -1).T

    def predict(block: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(block.sum(axis=1))

    chunked = predict_in_chunks(matrix, predict, chunk=7)
    assert np.allclose(chunked, matrix.sum(axis=1))


def test_predict_in_chunks_handles_empty() -> None:
    matrix = np.zeros((0, 4), dtype=np.float32)

    def predict(block: NDArray[Any]) -> NDArray[Any]:
        return np.zeros(block.shape[0], dtype=np.int64)

    assert predict_in_chunks(matrix, predict).shape == (0,)


def test_scatter_to_grid_places_values_at_mask() -> None:
    mask = np.array([[True, False], [False, True]])
    grid = scatter_to_grid(np.array([5, 9]), mask, fill=-1, dtype=np.int32)
    assert grid.tolist() == [[5, -1], [-1, 9]]


def test_scatter_to_grid_length_check() -> None:
    mask = np.array([[True, False], [True, True]])
    with pytest.raises(ValueError, match="values length"):
        scatter_to_grid(np.array([1.0]), mask)


def test_train_val_split_is_disjoint_and_deterministic() -> None:
    train, val = train_val_split(100, 0.2, seed=7)
    assert len(val) == 20
    assert len(train) == 80
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert sorted(train.tolist() + val.tolist()) == list(range(100))
    # Deterministic for a fixed seed.
    train2, val2 = train_val_split(100, 0.2, seed=7)
    assert np.array_equal(val, val2) and np.array_equal(train, train2)


def test_train_val_split_keeps_both_parts_nonempty() -> None:
    train, val = train_val_split(4, 0.01, seed=0)  # rounds to 0 -> clamped to 1
    assert len(val) == 1
    assert len(train) == 3


def test_train_val_split_validates() -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        train_val_split(10, 0.0)
    with pytest.raises(ValueError, match="n_samples"):
        train_val_split(0, 0.2)


def test_confusion_matrix_counts() -> None:
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


def test_confusion_matrix_validates_labels() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        confusion_matrix(np.array([0, 3]), np.array([0, 0]), 2)


def test_classification_metrics_perfect_and_partial() -> None:
    perfect = classification_metrics(np.eye(3) * 5)
    assert perfect["overall_accuracy"] == pytest.approx(1.0)
    assert np.allclose(perfect["f1"], 1.0)

    cm = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    metrics = classification_metrics(cm)
    assert metrics["overall_accuracy"] == pytest.approx(4 / 5)
    # Class 0: recall 1/2, precision 1/1 => f1 = 2*.5*1/(1.5).
    assert metrics["recall"][0] == pytest.approx(0.5)
    assert metrics["precision"][0] == pytest.approx(1.0)
    assert metrics["support"].tolist() == [2, 2, 1]


def test_classification_metrics_handles_empty_class() -> None:
    # Class 2 never appears (no support, no predictions) -> zeros, no divide error.
    cm = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=float)
    metrics = classification_metrics(cm)
    assert metrics["precision"][2] == 0.0
    assert metrics["recall"][2] == 0.0
    assert metrics["f1"][2] == 0.0


def test_select_best_index_max_min_and_ties() -> None:
    assert select_best_index([0.1, 0.5, 0.3], mode="max") == 1
    assert select_best_index([0.1, 0.5, 0.3], mode="min") == 0
    # Ties resolve to the first (simplest) candidate.
    assert select_best_index([0.5, 0.5, 0.2], mode="max") == 0
    with pytest.raises(ValueError, match="non-empty"):
        select_best_index([])


def test_format_accuracy_report_renders_metrics_and_matrix() -> None:
    cm = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=np.int64)
    metrics = classification_metrics(cm)
    report = format_accuracy_report(metrics, cm, ["water", "urban", "veg"], n_train=8, n_val=5)
    # Header facts.
    assert "held-out accuracy report" in report
    assert "Training samples: 8" in report
    assert "Validation samples: 5" in report
    assert "Overall accuracy: 0.8000" in report
    # Every class label appears, plus the confusion-matrix section.
    for label in ("water", "urban", "veg"):
        assert label in report
    assert "Confusion matrix (rows = true, columns = predicted):" in report
    # Trailing newline and one row per class in the matrix block.
    assert report.endswith("\n")


def test_format_accuracy_report_truncates_long_labels() -> None:
    cm = np.array([[2, 0], [0, 2]], dtype=np.int64)
    metrics = classification_metrics(cm)
    long_label = "a-very-long-class-name-that-overflows"
    report = format_accuracy_report(metrics, cm, [long_label, "other"], n_train=6, n_val=4)
    # The per-class row uses the full label; the matrix columns are truncated to 8.
    assert long_label in report
    assert long_label[:8] in report
