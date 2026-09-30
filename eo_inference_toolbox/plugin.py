"""Plugin entry point: registers the Processing provider and a launcher action."""

from __future__ import annotations

import os
from typing import Any

from qgis.core import QgsApplication

from .provider import EoInferenceProvider

_MENU = "&EO Inference Toolbox"


class EoInferencePlugin:
    """Thin QGIS plugin wrapper around the Processing provider and wizard dialog."""

    def __init__(self, iface: Any) -> None:
        self.iface = iface
        self.provider: EoInferenceProvider | None = None
        self._action: Any = None
        self._dialog: Any = None

    def initProcessing(self) -> None:
        self.provider = EoInferenceProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self) -> None:
        self.initProcessing()

        from qgis.PyQt.QtGui import QIcon
        from qgis.PyQt.QtWidgets import QAction

        icon = QIcon(os.path.join(os.path.dirname(__file__), "icon.svg"))
        self._action = QAction(icon, "EO Inference Toolbox", self.iface.mainWindow())
        self._action.setObjectName("eoInferenceToolboxAction")
        self._action.triggered.connect(self._run)
        self.iface.addPluginToMenu(_MENU, self._action)
        self.iface.addToolBarIcon(self._action)

    def unload(self) -> None:
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
        if self._action is not None:
            self.iface.removePluginMenu(_MENU, self._action)
            self.iface.removeToolBarIcon(self._action)
            # Sever the triggered->_run binding and delete the action so it does
            # not linger bound to this (now dead) plugin instance after a reload.
            self._action.triggered.disconnect(self._run)
            self._action.deleteLater()
            self._action = None
        if self._dialog is not None:
            # The modeless dialog is parented to the main window; close and
            # schedule deletion to avoid an orphaned window after reload.
            self._dialog.close()
            self._dialog.deleteLater()
            self._dialog = None

    def _run(self) -> None:
        from .gui.dialog import EoInferenceDialog

        if self._dialog is None:
            self._dialog = EoInferenceDialog(self.iface.mainWindow())
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()
