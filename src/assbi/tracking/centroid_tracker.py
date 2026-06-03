"""A lightweight multi-object tracker (greedy IoU + centroid association).

This is a SORT-style tracker without the Kalman filter / Hungarian dependency,
so it runs on pure Python. It is intentionally simple and well-commented: the
goal is correct, explainable identity assignment for line-crossing counts, not
benchmark-topping MOTA. For production-grade tracking the ``ObjectDetector``
output can instead be fed to Ultralytics' built-in ByteTrack — the rest of the
pipeline is unaffected because it only consumes :class:`Track` objects.
"""
from __future__ import annotations

from ..domain.models import Detection, Track


class CentroidTracker:
    """Associates per-frame detections into persistent tracks.

    Args:
        iou_threshold: minimum IoU for a detection to match an existing track.
        max_missed: frames a track may go unmatched before it is dropped.
        max_distance: fallback centroid distance (px) when IoU is zero but the
            objects are plausibly the same (fast motion / partial occlusion).
        history_length: number of past centroids retained for trail drawing.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 30,
        max_distance: float = 80.0,
        history_length: int = 32,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.history_length = history_length
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Advance the tracker by one frame and return the live tracks."""
        # Age every existing track; assume missed until matched this frame.
        for track in self._tracks.values():
            track.age += 1
            track.missed += 1

        matches = self._associate(detections)
        matched_det_ids: set[int] = set()

        for track_id, det_index in matches.items():
            det = detections[det_index]
            self._apply_match(self._tracks[track_id], det, frame_index)
            matched_det_ids.add(det_index)

        # Unmatched detections spawn new tracks.
        for i, det in enumerate(detections):
            if i not in matched_det_ids:
                self._spawn(det, frame_index)

        # Retire stale tracks.
        dead = [tid for tid, t in self._tracks.items() if t.missed > self.max_missed]
        for tid in dead:
            del self._tracks[tid]

        return self.tracks

    # -- internals ---------------------------------------------------------
    def _associate(self, detections: list[Detection]) -> dict[int, int]:
        """Greedy association: best (track, detection) pairs first.

        Returns a mapping of ``track_id -> detection_index``.
        """
        candidates: list[tuple[float, int, int]] = []  # (score, track_id, det_idx)
        for tid, track in self._tracks.items():
            for i, det in enumerate(detections):
                if det.object_class != track.object_class:
                    continue
                iou = track.box.iou(det.box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, tid, i))
                else:
                    dist = track.centroid.distance_to(det.centroid)
                    if iou == 0 and dist <= self.max_distance:
                        # Distance-based score in (0, iou_threshold) so IoU wins.
                        candidates.append((self.iou_threshold * (1 - dist / self.max_distance), tid, i))

        candidates.sort(reverse=True)  # highest score first
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matches: dict[int, int] = {}
        for _score, tid, det_idx in candidates:
            if tid in used_tracks or det_idx in used_dets:
                continue
            matches[tid] = det_idx
            used_tracks.add(tid)
            used_dets.add(det_idx)
        return matches

    def _apply_match(self, track: Track, det: Detection, frame_index: int) -> None:
        track.box = det.box
        track.confidence = det.confidence
        track.centroid = det.centroid
        track.last_seen_frame = frame_index
        track.hits += 1
        track.missed = 0
        track.history.append(det.centroid)
        if len(track.history) > self.history_length:
            track.history.pop(0)

    def _spawn(self, det: Detection, frame_index: int) -> None:
        track = Track(
            track_id=self._next_id,
            object_class=det.object_class,
            box=det.box,
            confidence=det.confidence,
            centroid=det.centroid,
            last_seen_frame=frame_index,
            history=[det.centroid],
        )
        self._tracks[self._next_id] = track
        self._next_id += 1
