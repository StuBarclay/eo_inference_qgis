"""The EO Inference processing algorithms and their registration helper.

``iter_algorithms`` is imported by :mod:`eo_inference_toolbox.provider` inside
``loadAlgorithms``, i.e. at algorithm-registration time. The concrete algorithm
modules import GDAL / scikit-learn lazily (inside ``processAlgorithm``), so
importing this package is cheap and safe even before those are installed.
"""

from __future__ import annotations

from collections.abc import Iterator

from .change import ChangeAlgorithm
from .classify import ClassifyAlgorithm
from .cluster import ClusterAlgorithm
from .embed import EmbedAlgorithm
from .fetch_imagery import FetchImageryAlgorithm
from .similarity import SimilarityAlgorithm
from .to_rgb import ToRgbAlgorithm

__all__ = [
    "ChangeAlgorithm",
    "ClassifyAlgorithm",
    "ClusterAlgorithm",
    "EmbedAlgorithm",
    "FetchImageryAlgorithm",
    "SimilarityAlgorithm",
    "ToRgbAlgorithm",
    "iter_algorithms",
]


def iter_algorithms() -> Iterator[object]:
    """Yield one fresh instance of each algorithm, in a sensible workflow order."""
    yield FetchImageryAlgorithm()
    yield EmbedAlgorithm()
    yield ToRgbAlgorithm()
    yield SimilarityAlgorithm()
    yield ChangeAlgorithm()
    yield ClassifyAlgorithm()
    yield ClusterAlgorithm()
