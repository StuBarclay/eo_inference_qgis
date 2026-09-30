"""Small numeric helpers shared across the pure cores."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_float(array: NDArray[Any]) -> NDArray[Any]:
    """Return ``array`` as a floating dtype, preserving float32 vs float64.

    Floating inputs pass through unchanged (so an int8 cube dequantised to
    float32 keeps its half-memory win); non-floating inputs are promoted to
    float64.
    """
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    out: NDArray[Any] = arr.astype(np.float64)
    return out
