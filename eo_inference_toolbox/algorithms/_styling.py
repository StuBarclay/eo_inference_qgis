"""Apply default renderers to output layers (continuous ramps and palettes).

These helpers are called from ``postProcessAlgorithm`` on the loaded output
layer. They fail soft: any styling error is swallowed so a valid raster is never
withheld from the user because of a cosmetic problem.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

from qgis.core import (
    QgsColorRampShader,
    QgsPalettedRasterRenderer,
    QgsRasterBandStats,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)
from qgis.PyQt.QtGui import QColor


def apply_continuous_ramp(
    layer: object, *, band: int = 1, low_color: str = "#2c7bb6", high_color: str = "#d7191c"
) -> None:
    """Blue -> yellow -> red stretch across the band's actual min/max."""
    try:
        provider = layer.dataProvider()  # type: ignore[attr-defined]
        stats = provider.bandStatistics(band, QgsRasterBandStats.All)
        vmin, vmax = float(stats.minimumValue), float(stats.maximumValue)
        if not vmax > vmin:
            return
        mid = 0.5 * (vmin + vmax)
        ramp = [
            QgsColorRampShader.ColorRampItem(vmin, QColor(low_color), f"{vmin:.3g}"),
            QgsColorRampShader.ColorRampItem(mid, QColor("#ffffbf"), f"{mid:.3g}"),
            QgsColorRampShader.ColorRampItem(vmax, QColor(high_color), f"{vmax:.3g}"),
        ]
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        color_ramp.setColorRampItemList(ramp)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(color_ramp)
        renderer = QgsSingleBandPseudoColorRenderer(provider, band, shader)
        layer.setRenderer(renderer)  # type: ignore[attr-defined]
        layer.triggerRepaint()  # type: ignore[attr-defined]
    except Exception:
        return


def _class_color(index: int, total: int) -> QColor:
    hue = (index / max(total, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def apply_paletted(layer: object, values: Sequence[int], *, band: int = 1) -> None:
    """Assign a distinct colour to each integer class value."""
    try:
        provider = layer.dataProvider()  # type: ignore[attr-defined]
        ordered = sorted({int(v) for v in values})
        classes = []
        for index, value in enumerate(ordered):
            klass = QgsPalettedRasterRenderer.Class(
                value, _class_color(index, len(ordered)), str(value)
            )
            classes.append(klass)
        renderer = QgsPalettedRasterRenderer(provider, band, classes)
        layer.setRenderer(renderer)  # type: ignore[attr-defined]
        layer.triggerRepaint()  # type: ignore[attr-defined]
    except Exception:
        return
