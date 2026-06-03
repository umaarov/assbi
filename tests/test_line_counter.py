from assbi.analytics.line_counter import LineCounter
from assbi.domain.geometry import BoundingBox, CountingLine, Point
from assbi.domain.models import ObjectClass, Track


def _track(track_id, cls, cx, cy):
    box = BoundingBox(cx - 5, cy - 5, cx + 5, cy + 5)
    return Track(
        track_id=track_id, object_class=cls, box=box,
        confidence=0.9, centroid=Point(cx, cy), last_seen_frame=0,
    )


def _line():
    return CountingLine(Point(0, 100), Point(200, 100), "mid")


def test_person_crossing_counted_once():
    counter = LineCounter([_line()])
    # frame 0: above the line (establishes prev centroid)
    counter.process([_track(1, ObjectClass.PERSON, 50, 80)], 0)
    # frame 1: crosses below -> IN
    events = counter.process([_track(1, ObjectClass.PERSON, 50, 120)], 1)
    assert len(events) == 1
    assert counter.people_in == 1
    # staying below should not double count
    counter.process([_track(1, ObjectClass.PERSON, 50, 130)], 2)
    assert counter.people_in == 1


def test_vehicle_and_person_separated():
    counter = LineCounter([_line()])
    counter.process([_track(1, ObjectClass.CAR, 50, 80)], 0)
    counter.process([_track(1, ObjectClass.CAR, 50, 120)], 1)
    assert counter.vehicles_in == 1
    assert counter.people_in == 0


def test_out_direction():
    counter = LineCounter([_line()])
    counter.process([_track(2, ObjectClass.PERSON, 50, 120)], 0)
    counter.process([_track(2, ObjectClass.PERSON, 50, 80)], 1)
    assert counter.people_out == 1
    assert counter.people_in == 0


def test_breakdown_structure():
    counter = LineCounter([_line()])
    counter.process([_track(1, ObjectClass.PERSON, 50, 80)], 0)
    counter.process([_track(1, ObjectClass.PERSON, 50, 120)], 1)
    bd = counter.breakdown()
    assert bd["mid"]["in"]["person"] == 1
