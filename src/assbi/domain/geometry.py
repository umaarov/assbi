"""Pure-geometry value objects used across the platform.

These types are deliberately framework-free (no numpy / OpenCV) so that the
domain layer can run, and be unit-tested, in any environment. Heavier adapters
(YOLO, OpenCV) convert *to* and *from* these types at the boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class Point:
    """A 2-D point in pixel coordinates (origin top-left, y grows downward)."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[int, int]:
        return int(round(self.x)), int(round(self.y))


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union — the standard association metric."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class CrossingDirection(str, Enum):
    """Which way a track moved across the counting line."""

    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class CountingLine:
    """A virtual line segment used to count objects that cross it.

    Crossing direction is derived from the signed side of the line: a track
    moving from the negative half-plane to the positive one is counted as ``IN``
    and the reverse as ``OUT``. The mapping is purely a convention and is fixed
    so reports are stable.
    """

    start: Point
    end: Point
    name: str = "line"

    def side(self, p: Point) -> float:
        """Signed area (2-D cross product); sign indicates the half-plane."""
        return ((self.end.x - self.start.x) * (p.y - self.start.y)
                - (self.end.y - self.start.y) * (p.x - self.start.x))

    def crossing(self, previous: Point, current: Point) -> CrossingDirection | None:
        """Return the direction if the segment prev->current crosses the line.

        Uses a robust segment-segment intersection test so that fast-moving
        objects (large jumps between frames) are still counted, then resolves
        direction from the change in signed side.
        """
        if not _segments_intersect(self.start, self.end, previous, current):
            return None
        before, after = self.side(previous), self.side(current)
        if before == after:  # grazing / collinear — ignore to avoid double counts
            return None
        return CrossingDirection.IN if after > before else CrossingDirection.OUT


def _orientation(a: Point, b: Point, c: Point) -> int:
    val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (min(a.x, c.x) <= b.x <= max(a.x, c.x)
            and min(a.y, c.y) <= b.y <= max(a.y, c.y))


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Classic CLRS segment-intersection predicate (handles collinear cases)."""
    o1, o2 = _orientation(p1, p2, p3), _orientation(p1, p2, p4)
    o3, o4 = _orientation(p3, p4, p1), _orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False
