"""Core domain entities for the surveillance analytics pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .geometry import BoundingBox, CrossingDirection, Point


class ObjectClass(str, Enum):
    """Classes the platform tracks. Maps onto the COCO labels YOLO emits."""

    PERSON = "person"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"

    @property
    def is_vehicle(self) -> bool:
        return self in {
            ObjectClass.CAR,
            ObjectClass.TRUCK,
            ObjectClass.BUS,
            ObjectClass.MOTORCYCLE,
        }

    @classmethod
    def from_label(cls, label: str) -> "ObjectClass | None":
        try:
            return cls(label.lower())
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detected in a single frame (pre-tracking)."""

    object_class: ObjectClass
    box: BoundingBox
    confidence: float

    @property
    def centroid(self) -> Point:
        return self.box.centroid


@dataclass(slots=True)
class Track:
    """A detection associated across frames into a persistent identity."""

    track_id: int
    object_class: ObjectClass
    box: BoundingBox
    confidence: float
    centroid: Point
    last_seen_frame: int
    age: int = 0          # number of frames this track has existed
    hits: int = 1         # number of frames it was actually matched
    missed: int = 0       # consecutive frames without a match
    counted: bool = False  # whether it has already been counted by a line
    history: list[Point] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """Emitted when a tracked object crosses a counting line."""

    track_id: int
    object_class: ObjectClass
    direction: CrossingDirection
    line_name: str
    frame_index: int
    timestamp: datetime

    @staticmethod
    def now(**kwargs) -> "CrossingEvent":
        return CrossingEvent(timestamp=datetime.now(timezone.utc), **kwargs)


@dataclass(slots=True)
class FrameAnalytics:
    """Per-frame snapshot persisted as the structured analytics time-series."""

    frame_index: int
    timestamp: datetime
    person_count: int            # live persons in frame (crowd density)
    vehicle_count: int           # live vehicles in frame
    total_detections: int
    crossings_in: int            # cumulative
    crossings_out: int           # cumulative
    is_anomaly: bool = False
    anomaly_score: float = 0.0


@dataclass(slots=True)
class SessionSummary:
    """Roll-up KPIs for a completed analysis session."""

    session_id: str
    source: str
    frames_processed: int
    duration_seconds: float
    people_in: int
    people_out: int
    vehicles_in: int
    vehicles_out: int
    peak_crowd: int
    peak_crowd_frame: int
    anomalies: int
    avg_confidence: float

    @property
    def net_people(self) -> int:
        return self.people_in - self.people_out

    @property
    def total_crossings(self) -> int:
        return self.people_in + self.people_out + self.vehicles_in + self.vehicles_out
