"""QGIS Processing provider for the EO Inference Toolbox."""

from __future__ import annotations

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from . import __version__

PROVIDER_ID = "eoinfer"
PROVIDER_NAME = "EO Inference Toolbox"


class EoInferenceProvider(QgsProcessingProvider):
    """Registers the toolbox algorithms under the ``eoinfer:`` namespace."""

    def id(self) -> str:
        return PROVIDER_ID

    def name(self) -> str:
        return PROVIDER_NAME

    def longName(self) -> str:
        return f"{PROVIDER_NAME} {__version__}"

    def icon(self) -> QIcon:
        return QIcon(os.path.join(os.path.dirname(__file__), "icon.svg"))

    def loadAlgorithms(self) -> None:
        # Imported lazily so provider construction never drags in numpy/GDAL.
        from .algorithms import iter_algorithms

        for algorithm in iter_algorithms():
            self.addAlgorithm(algorithm)
