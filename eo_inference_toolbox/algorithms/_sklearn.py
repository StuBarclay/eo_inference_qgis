"""Optional scikit-learn tier: detect, offer a guided install, build estimators.

scikit-learn is never a hard dependency and never vendored. The classify and
cluster algorithms call :func:`ensure_available` first, which raises a clear,
actionable :class:`QgsProcessingException` (with a copy-pasteable pip command)
when it is missing. The estimator factories keep the sklearn import inside the
functions, so importing this module is cheap and safe.
"""

from __future__ import annotations

import sys
from typing import Any

from ..eocore.backends._deps import module_available

CLASSIFIERS: tuple[str, ...] = ("random_forest", "gradient_boosting")


def is_available() -> bool:
    """True if scikit-learn is importable (delegates to the shared detector)."""
    return module_available("sklearn")


def install_hint() -> str:
    """A copy-pasteable pip command for scikit-learn."""
    return f'"{sys.executable}" -m pip install scikit-learn'


def ensure_available() -> None:
    """Raise a QGIS-friendly error if scikit-learn is absent."""
    if is_available():
        return
    from qgis.core import QgsProcessingException

    raise QgsProcessingException(
        "This algorithm needs scikit-learn, which is not installed.\n"
        f"Install it from the OSGeo4W/QGIS Python shell with:\n  {install_hint()}\n"
        "or use the 'Install scikit-learn' button in the EO Inference Toolbox dialog."
    )


def make_classifier(algorithm: str, *, n_estimators: int = 200, seed: int = 0) -> Any:
    """Build an unfitted classifier. Imports sklearn lazily."""
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(n_estimators), random_state=int(seed), n_jobs=-1
        )
    if algorithm == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=int(seed))
    raise ValueError(f"unknown classifier {algorithm!r}; expected one of {CLASSIFIERS}")


def make_kmeans(n_clusters: int, *, seed: int = 0) -> Any:
    """Build an unfitted KMeans estimator. Imports sklearn lazily."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=10)


def make_minibatch_kmeans(n_clusters: int, *, seed: int = 0, batch_size: int = 4096) -> Any:
    """Build an unfitted MiniBatchKMeans estimator. Imports sklearn lazily.

    Used by the opt-in low-memory cluster path, which streams the whole raster
    through :meth:`partial_fit` (see
    :func:`eo_inference_toolbox.eocore.streaming.partial_fit_stream`) so the fit
    sees *every* valid pixel rather than a subsample. ``n_init`` is a small,
    explicit integer to stay reproducible and quiet across scikit-learn versions
    (whose ``"auto"`` default has drifted).
    """
    from sklearn.cluster import MiniBatchKMeans

    return MiniBatchKMeans(
        n_clusters=int(n_clusters),
        random_state=int(seed),
        n_init=3,
        batch_size=int(batch_size),
    )


def silhouette(samples: Any, labels: Any) -> float:
    """Mean silhouette coefficient for a labelled sample. Imports sklearn lazily.

    Used by the opt-in auto-K cluster search to rank candidate K values; higher
    is better. The caller is responsible for capping ``samples`` to a size where
    the O(n^2) computation is affordable.
    """
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(samples, labels))
