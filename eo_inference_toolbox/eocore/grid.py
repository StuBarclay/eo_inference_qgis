"""Raster grid / affine geometry, pure stdlib (no GDAL)."""

from __future__ import annotations

import math
from dataclasses import dataclass

# (min_x, min_y, max_x, max_y)
Bounds = tuple[float, float, float, float]


def bboxes_intersect(a: Bounds, b: Bounds) -> bool:
    """True if two axis-aligned bounding boxes overlap (touching edges excluded)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


@dataclass(frozen=True)
class GridSpec:
    """A north-up raster grid: top-left corner, square pixel, and dimensions."""

    min_x: float
    max_y: float
    pixel_size: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-style geotransform ``(originX, dx, 0, originY, 0, -dy)``."""
        return (self.min_x, self.pixel_size, 0.0, self.max_y, 0.0, -self.pixel_size)

    @property
    def bounds(self) -> Bounds:
        max_x = self.min_x + self.width * self.pixel_size
        min_y = self.max_y - self.height * self.pixel_size
        return (self.min_x, min_y, max_x, self.max_y)

    def xy_center(self, row: int, col: int) -> tuple[float, float]:
        """World coordinates of the centre of pixel (row, col)."""
        x = self.min_x + (col + 0.5) * self.pixel_size
        y = self.max_y - (row + 0.5) * self.pixel_size
        return (x, y)

    def rowcol_for_xy(self, x: float, y: float) -> tuple[int, int]:
        """(row, col) of the pixel containing world coordinate (x, y)."""
        col = math.floor((x - self.min_x) / self.pixel_size)
        row = math.floor((self.max_y - y) / self.pixel_size)
        return (row, col)

    def contains_rowcol(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


def grid_from_bounds(bounds: Bounds, pixel_size: float) -> GridSpec:
    """Build a grid covering ``bounds`` at ``pixel_size`` (ceil to cover fully)."""
    min_x, min_y, max_x, max_y = bounds
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("bounds must have positive extent")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    width = max(1, math.ceil((max_x - min_x) / pixel_size))
    height = max(1, math.ceil((max_y - min_y) / pixel_size))
    return GridSpec(min_x=min_x, max_y=max_y, pixel_size=pixel_size, width=width, height=height)


# Mean metres per degree of latitude on the WGS84 ellipsoid (the meridional value
# runs ~110.57 km at the equator to ~111.69 km at the poles). Good enough to turn a
# backend's native metre resolution into a sensible default pixel size when the user
# picks a geographic (degree) CRS.
_METRES_PER_DEGREE = 111_320.0


def metres_to_degrees(metres: float) -> float:
    """Approximate the degree-equivalent of a metre distance (WGS84, ~111.32 km/deg).

    Used to pick a default pixel size in a geographic CRS from a backend's native
    metre resolution: using the metre count verbatim as a degree pixel size would
    build a grossly oversized grid (e.g. a 10 m resolution becoming 10 degrees --
    over a thousand kilometres -- per pixel). This is a latitude-independent
    approximation (it ignores the ``cos(latitude)`` shrinkage of a degree of
    longitude), so it is a reasonable default, not a projection.
    """
    if metres <= 0:
        raise ValueError("metres must be positive")
    return metres / _METRES_PER_DEGREE
