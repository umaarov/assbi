from assbi.domain.geometry import (
    BoundingBox,
    CountingLine,
    CrossingDirection,
    Point,
)


def test_bbox_centroid_and_area():
    box = BoundingBox(0, 0, 10, 20)
    assert box.area == 200
    assert box.centroid == Point(5, 10)


def test_iou_identical_boxes():
    a = BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0


def test_iou_disjoint_boxes():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert a.iou(b) == 0.0


def test_iou_half_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 0, 15, 10)
    # intersection 50, union 150
    assert abs(a.iou(b) - (50 / 150)) < 1e-9


def test_line_crossing_in_direction():
    line = CountingLine(Point(0, 100), Point(200, 100), "mid")
    # moving downward across the line
    assert line.crossing(Point(100, 90), Point(100, 110)) == CrossingDirection.IN


def test_line_crossing_out_direction():
    line = CountingLine(Point(0, 100), Point(200, 100), "mid")
    assert line.crossing(Point(100, 110), Point(100, 90)) == CrossingDirection.OUT


def test_no_crossing_when_same_side():
    line = CountingLine(Point(0, 100), Point(200, 100), "mid")
    assert line.crossing(Point(100, 80), Point(100, 90)) is None
