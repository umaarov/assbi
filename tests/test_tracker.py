from assbi.domain.geometry import BoundingBox
from assbi.domain.models import Detection, ObjectClass
from assbi.tracking.centroid_tracker import CentroidTracker


def _det(cls, cx, cy, size=10):
    return Detection(cls, BoundingBox(cx - size, cy - size, cx + size, cy + size), 0.9)


def test_track_persists_across_frames():
    tracker = CentroidTracker()
    t0 = tracker.update([_det(ObjectClass.PERSON, 100, 100)], 0)
    assert len(t0) == 1
    tid = t0[0].track_id
    # small move -> should keep the same id
    t1 = tracker.update([_det(ObjectClass.PERSON, 105, 102)], 1)
    assert len(t1) == 1
    assert t1[0].track_id == tid
    assert t1[0].hits == 2


def test_new_object_gets_new_id():
    tracker = CentroidTracker()
    tracker.update([_det(ObjectClass.PERSON, 100, 100)], 0)
    tracks = tracker.update(
        [_det(ObjectClass.PERSON, 105, 100), _det(ObjectClass.PERSON, 500, 500)], 1
    )
    assert len({t.track_id for t in tracks}) == 2


def test_stale_track_is_dropped():
    tracker = CentroidTracker(max_missed=2)
    tracker.update([_det(ObjectClass.PERSON, 100, 100)], 0)
    for f in range(1, 6):
        tracker.update([], f)
    assert tracker.tracks == []


def test_class_mismatch_not_associated():
    tracker = CentroidTracker()
    tracker.update([_det(ObjectClass.PERSON, 100, 100)], 0)
    tracks = tracker.update([_det(ObjectClass.CAR, 100, 100)], 1)
    # person went missing, a new car track was created
    classes = {t.object_class for t in tracks}
    assert ObjectClass.CAR in classes
