"""Pure-Python compute cores for the EO Inference Toolbox.

Nothing in this package imports QGIS, GDAL or PyTorch at module load time, so the
whole subtree is unit-testable in a plain Python + NumPy environment. The heavy,
backend-specific machinery lives behind lazy imports in ``eocore.backends``.
"""

from __future__ import annotations
