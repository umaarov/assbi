"""Framework-free domain layer: entities, value objects and ports."""
from .geometry import (
    BoundingBox,
    CountingLine,
    CrossingDirection,
    Point,
)
from .interfaces import (
    AnalyticsRepository,
    Frame,
    ObjectDetector,
    VideoSource,
)
from .models import (
    CrossingEvent,
    Detection,
    FrameAnalytics,
    ObjectClass,
    SessionSummary,
    Track,
)

__all__ = [
    "BoundingBox",
    "CountingLine",
    "CrossingDirection",
    "Point",
    "AnalyticsRepository",
    "Frame",
    "ObjectDetector",
    "VideoSource",
    "CrossingEvent",
    "Detection",
    "FrameAnalytics",
    "ObjectClass",
    "SessionSummary",
    "Track",
]
