"""Virtual line-crossing counter — the headline feature.

Given a stream of tracked objects, it counts how many persons and vehicles
cross one or more virtual lines, broken down by direction (in/out). This is the
"draw a line on the video and count people and cars crossing it" requirement.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from ..domain.geometry import CountingLine, CrossingDirection
from ..domain.models import CrossingEvent, ObjectClass, Track


class LineCounter:
    """Counts tracked objects crossing a set of named virtual lines.

    A track is counted at most once *per line per direction transition*: we
    compare its previous centroid with the current one and look for a segment
    intersection, which makes the count robust to jitter and frame skips.
    """

    def __init__(self, lines: list[CountingLine]) -> None:
        if not lines:
            raise ValueError("At least one counting line is required.")
        self.lines = lines
        # counts[line_name][direction][class] -> int
        self._counts: dict[str, dict[CrossingDirection, dict[ObjectClass, int]]] = {
            line.name: {
                CrossingDirection.IN: defaultdict(int),
                CrossingDirection.OUT: defaultdict(int),
            }
            for line in lines
        }
        self._prev_centroid: dict[int, "tuple[float, float]"] = {}

    def process(self, tracks: list[Track], frame_index: int) -> list[CrossingEvent]:
        """Detect crossings for this frame and return the emitted events."""
        events: list[CrossingEvent] = []
        live_ids = {t.track_id for t in tracks}

        for track in tracks:
            prev = self._prev_centroid.get(track.track_id)
            current = track.centroid
            if prev is not None:
                from ..domain.geometry import Point

                prev_point = Point(prev[0], prev[1])
                for line in self.lines:
                    direction = line.crossing(prev_point, current)
                    if direction is not None:
                        self._counts[line.name][direction][track.object_class] += 1
                        events.append(
                            CrossingEvent(
                                track_id=track.track_id,
                                object_class=track.object_class,
                                direction=direction,
                                line_name=line.name,
                                frame_index=frame_index,
                                timestamp=datetime.now(timezone.utc),
                            )
                        )
                        track.counted = True
            self._prev_centroid[track.track_id] = (current.x, current.y)

        # Forget centroids of tracks that have disappeared (free memory).
        for tid in list(self._prev_centroid):
            if tid not in live_ids:
                del self._prev_centroid[tid]

        return events

    # -- aggregate queries -------------------------------------------------
    def count(
        self,
        direction: CrossingDirection,
        *,
        line_name: str | None = None,
        vehicles: bool = False,
    ) -> int:
        """Total crossings for a direction, optionally filtered by line/type."""
        total = 0
        for name, by_dir in self._counts.items():
            if line_name and name != line_name:
                continue
            for cls, n in by_dir[direction].items():
                if vehicles and not cls.is_vehicle:
                    continue
                if not vehicles and cls.is_vehicle:
                    continue
                total += n
        return total

    @property
    def people_in(self) -> int:
        return self.count(CrossingDirection.IN, vehicles=False)

    @property
    def people_out(self) -> int:
        return self.count(CrossingDirection.OUT, vehicles=False)

    @property
    def vehicles_in(self) -> int:
        return self.count(CrossingDirection.IN, vehicles=True)

    @property
    def vehicles_out(self) -> int:
        return self.count(CrossingDirection.OUT, vehicles=True)

    def breakdown(self) -> dict:
        """Nested dict of all counts — handy for reports and the dashboard."""
        return {
            name: {
                direction.value: {cls.value: n for cls, n in classes.items()}
                for direction, classes in by_dir.items()
            }
            for name, by_dir in self._counts.items()
        }
