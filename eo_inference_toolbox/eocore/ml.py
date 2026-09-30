"""Pure NumPy helpers shared by the classify / cluster algorithms.

These functions own no scikit-learn: the estimator is injected as a ``predict``
callable, so all the reshaping / masking / scattering logic is unit-testable
without sklearn installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._num import as_float

Predict = Callable[[NDArray[Any]], Any]


def cube_to_matrix(cube: NDArray[Any]) -> NDArray[Any]:
    """Reshape a ``(bands, H, W)`` cube to a ``(H*W, bands)`` sample matrix."""
    arr = as_float(cube)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (bands, H, W) cube, got shape {arr.shape}")
    bands, height, width = arr.shape
    out: NDArray[Any] = arr.reshape(bands, height * width).T
    return out


def valid_mask_from_nodata(cube: NDArray[Any], nodata: float) -> NDArray[Any]:
    """Boolean ``(H, W)`` mask of pixels that are finite and not all-nodata.

    A pixel counts as valid when every band is finite **and** at least one band
    differs from ``nodata``. This is the deliberate "all bands" rule shared with
    :func:`eo_inference_toolbox.eocore.runner.input_valid_mask`: a pixel is only
    dropped as fill when *every* band equals the sentinel, so a real embedding
    vector that merely happens to hit ``nodata`` in one or two components is
    kept. A partially-nodata pixel (some bands sentinel, some not) is therefore
    treated as valid; embedding cubes stamp the sentinel across all bands
    together (see :func:`~eo_inference_toolbox.eocore.runner.finalize_embedding`),
    so this case does not arise for our own outputs.
    """
    arr = np.asarray(cube)
    finite = np.all(np.isfinite(arr), axis=0)
    not_fill = np.any(arr != nodata, axis=0)
    out: NDArray[Any] = finite & not_fill
    return out


def encode_labels(labels: Sequence[Any]) -> tuple[NDArray[Any], list[Any]]:
    """Map arbitrary labels to contiguous int codes, preserving first-seen order.

    Returns ``(codes, classes)`` where ``classes[code]`` recovers the label.
    """
    classes: list[Any] = []
    seen: dict[Any, int] = {}
    codes = np.empty(len(labels), dtype=np.int64)
    for index, label in enumerate(labels):
        if label not in seen:
            seen[label] = len(classes)
            classes.append(label)
        codes[index] = seen[label]
    return codes, classes


def predict_in_chunks(
    matrix: NDArray[Any], predict: Predict, *, chunk: int = 100_000
) -> NDArray[Any]:
    """Apply ``predict`` to a large matrix in row chunks and concatenate results."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    n_rows = matrix.shape[0]
    if n_rows == 0:
        empty: NDArray[Any] = np.asarray(predict(matrix[:0]))
        return empty
    parts: list[NDArray[Any]] = []
    for start in range(0, n_rows, chunk):
        block = matrix[start : start + chunk]
        parts.append(np.asarray(predict(block)))
    out: NDArray[Any] = np.concatenate(parts, axis=0)
    return out


def scatter_to_grid(
    values: NDArray[Any],
    mask: NDArray[Any],
    *,
    fill: float = 0.0,
    dtype: Any = np.float32,
) -> NDArray[Any]:
    """Scatter a 1-D ``values`` array back onto a ``(H, W)`` grid at ``mask`` pixels."""
    mask_arr = np.asarray(mask, dtype=bool)
    flat_mask = mask_arr.reshape(-1)
    n_true = int(np.count_nonzero(flat_mask))
    vals = np.asarray(values)
    if vals.shape[0] != n_true:
        raise ValueError(f"values length {vals.shape[0]} != number of True mask pixels {n_true}")
    grid = np.full(flat_mask.shape, fill, dtype=dtype)
    grid[flat_mask] = vals.astype(dtype, copy=False)
    out: NDArray[Any] = grid.reshape(mask_arr.shape)
    return out


def train_val_split(
    n_samples: int, val_fraction: float, *, seed: int = 0
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return ``(train_idx, val_idx)`` for a deterministic shuffle split.

    ``val_fraction`` is clamped so both parts are non-empty when ``n_samples>=2``.
    Used only when the caller opts into a held-out accuracy report; the default
    classify path trains on everything.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = round(n_samples * val_fraction)
    n_val = max(1, min(n_val, n_samples - 1)) if n_samples >= 2 else 0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def confusion_matrix(y_true: NDArray[Any], y_pred: NDArray[Any], n_classes: int) -> NDArray[Any]:
    """Row=true, col=predicted integer-coded confusion matrix (``n_classes`` square)."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    yt = np.asarray(y_true).astype(np.int64).reshape(-1)
    yp = np.asarray(y_pred).astype(np.int64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if yt.size and (yt.min() < 0 or yt.max() >= n_classes or yp.min() < 0 or yp.max() >= n_classes):
        raise ValueError("labels must be in [0, n_classes)")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def classification_metrics(cm: NDArray[Any]) -> dict[str, Any]:
    """Overall accuracy and per-class precision/recall/F1/support from a matrix.

    Divisions guard against empty rows/columns (a class with no support or no
    predictions scores 0 rather than raising).
    """
    matrix = np.asarray(cm, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square")
    total = matrix.sum()
    correct = float(np.trace(matrix))
    overall = correct / total if total > 0 else 0.0
    tp = np.diag(matrix)
    pred_tot = matrix.sum(axis=0)
    true_tot = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        recall = np.where(true_tot > 0, tp / true_tot, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    return {
        "overall_accuracy": overall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_tot.astype(np.int64),
    }


def select_best_index(scores: Sequence[float], *, mode: str = "max") -> int:
    """Index of the best score. ``mode='max'`` (e.g. silhouette) or ``'min'``.

    Ties resolve to the first (lowest-index) candidate, so for auto-k selection
    the simpler model wins when scores are equal.
    """
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        raise ValueError("scores must be non-empty")
    if mode == "max":
        return int(np.argmax(values))
    if mode == "min":
        return int(np.argmin(values))
    raise ValueError(f"unknown mode {mode!r}; expected 'max' or 'min'")


def format_accuracy_report(
    metrics: dict[str, Any],
    cm: NDArray[Any],
    classes: Sequence[Any],
    n_train: int,
    n_val: int,
) -> str:
    """Render a plain-text held-out accuracy report from metrics + confusion matrix.

    Pure (no QGIS / sklearn), so the classify algorithm's report output is
    unit-testable. ``metrics`` is the dict from :func:`classification_metrics`.
    """
    labels = list(classes)
    lines = [
        "EO Inference Toolbox - held-out accuracy report",
        "=" * 48,
        f"Training samples: {n_train}",
        f"Validation samples: {n_val}",
        f"Overall accuracy: {metrics['overall_accuracy']:.4f}",
        "",
        f"{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = metrics["support"]
    for index, label in enumerate(labels):
        lines.append(
            f"{label!s:<20}{precision[index]:>10.3f}{recall[index]:>10.3f}"
            f"{f1[index]:>10.3f}{int(support[index]):>10}"
        )
    lines += ["", "Confusion matrix (rows = true, columns = predicted):"]
    header = "".join(f"{str(label)[:8]:>10}" for label in labels)
    lines.append(f"{'':<12}{header}")
    for index, label in enumerate(labels):
        row = "".join(f"{int(v):>10}" for v in cm[index])
        lines.append(f"{str(label)[:10]:<12}{row}")
    return "\n".join(lines) + "\n"
