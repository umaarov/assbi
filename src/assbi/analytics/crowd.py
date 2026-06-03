"""Crowd density analytics derived from per-frame track populations."""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import ObjectClass, Track


@dataclass(frozen=True, slots=True)
class CrowdSnapshot:
    """Live population of a single frame."""

    person_count: int
    vehicle_count: int
    total: int
    density_level: str  # "low" | "moderate" | "high" | "critical"


class CrowdAnalyzer:
    """Classifies live crowd density using configurable occupancy thresholds.

    Thresholds are expressed as person counts; tune them per-scene (a corridor
    saturates far sooner than a stadium concourse).
    """

    def __init__(
        self,
        moderate: int = 8,
        high: int = 20,
        critical: int = 40,
    ) -> None:
        self.moderate = moderate
        self.high = high
        self.critical = critical

    def snapshot(self, tracks: list[Track]) -> CrowdSnapshot:
        persons = sum(1 for t in tracks if t.object_class == ObjectClass.PERSON)
        vehicles = sum(1 for t in tracks if t.object_class.is_vehicle)
        return CrowdSnapshot(
            person_count=persons,
            vehicle_count=vehicles,
            total=len(tracks),
            density_level=self._level(persons),
        )

    def _level(self, persons: int) -> str:
        if persons >= self.critical:
            return "critical"
        if persons >= self.high:
            return "high"
        if persons >= self.moderate:
            return "moderate"
        return "low"
