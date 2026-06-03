"""Draws detections, tracks, counting lines and live KPIs onto frames.

All OpenCV usage is isolated here. If a frame carries no image (headless
simulation) the methods are no-ops, so the orchestrator code stays uniform.
"""
from __future__ import annotations

from ..analytics.crowd import CrowdSnapshot
from ..analytics.line_counter import LineCounter
from ..domain.geometry import CountingLine
from ..domain.interfaces import Frame
from ..domain.models import ObjectClass, Track

_CLASS_COLOR = {
    ObjectClass.PERSON: (0, 200, 0),
    ObjectClass.CAR: (0, 160, 255),
    ObjectClass.TRUCK: (0, 120, 220),
    ObjectClass.BUS: (0, 90, 200),
    ObjectClass.MOTORCYCLE: (40, 200, 255),
    ObjectClass.BICYCLE: (200, 200, 0),
}


class FrameAnnotator:
    def __init__(
        self,
        lines: list[CountingLine],
        privacy_mode: str = "off",
        privacy_targets: list[str] | None = None,
        privacy_strength: int = 23,
    ) -> None:
        self._lines = lines
        self._privacy_mode = (privacy_mode or "off").lower()
        self._privacy_targets = set(privacy_targets or ["person"])
        self._privacy_strength = max(3, int(privacy_strength))
        try:
            import cv2  # noqa: WPS433

            self._cv2 = cv2
        except ImportError:  # pragma: no cover - environment dependent
            self._cv2 = None

    @property
    def available(self) -> bool:
        return self._cv2 is not None

    def draw(
        self,
        frame: Frame,
        tracks: list[Track],
        counter: LineCounter,
        crowd: CrowdSnapshot,
    ) -> None:  # pragma: no cover - requires cv2 + image
        if self._cv2 is None or frame.image is None:
            return
        cv2 = self._cv2
        img = frame.image

        # Privacy first: anonymise people/targets *before* drawing overlays, so
        # boxes and trails remain visible on top of the blurred regions.
        if self._privacy_mode != "off":
            self.anonymize(img, tracks)

        for line in self._lines:
            cv2.line(img, line.start.as_tuple(), line.end.as_tuple(), (0, 0, 255), 2)
            cv2.putText(img, line.name, line.start.as_tuple(),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for t in tracks:
            color = _CLASS_COLOR.get(t.object_class, (255, 255, 255))
            x1, y1 = t.box.x1, t.box.y1
            x2, y2 = t.box.x2, t.box.y2
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{t.object_class.value} #{t.track_id}",
                        (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            # Motion trail.
            for a, b in zip(t.history, t.history[1:]):
                cv2.line(img, a.as_tuple(), b.as_tuple(), color, 1)

        self._draw_hud(img, counter, crowd)

    def anonymize(self, img, tracks: list[Track]) -> None:
        """Blur or pixelate the bounding box of each privacy-target track.

        Anonymises identities (GDPR data-minimisation) while preserving the
        analytics: counts and tracks are unaffected because detection already
        happened upstream — only the displayed pixels are obscured.
        """
        if self._cv2 is None or img is None:
            return
        cv2 = self._cv2
        h, w = img.shape[:2]
        k = self._privacy_strength | 1  # kernel size must be odd
        for t in tracks:
            if t.object_class.value not in self._privacy_targets:
                continue
            x1 = max(0, int(t.box.x1)); y1 = max(0, int(t.box.y1))
            x2 = min(w, int(t.box.x2)); y2 = min(h, int(t.box.y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            roi = img[y1:y2, x1:x2]
            if self._privacy_mode == "pixelate":
                blocks = max(2, (x2 - x1) // self._privacy_strength)
                small = cv2.resize(roi, (blocks, max(2, (y2 - y1) * blocks // (x2 - x1))),
                                   interpolation=cv2.INTER_LINEAR)
                roi = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            else:  # "blur"
                roi = cv2.GaussianBlur(roi, (k, k), 0)
            img[y1:y2, x1:x2] = roi

    def _draw_hud(self, img, counter: LineCounter, crowd: CrowdSnapshot) -> None:  # pragma: no cover
        cv2 = self._cv2
        lines = [
            f"People IN: {counter.people_in}   OUT: {counter.people_out}",
            f"Vehicles IN: {counter.vehicles_in}   OUT: {counter.vehicles_out}",
            f"Live crowd: {crowd.person_count} ({crowd.density_level})",
        ]
        cv2.rectangle(img, (0, 0), (360, 80), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter that no-ops without OpenCV."""

    def __init__(self, path: str, fps: float, frame_size: tuple[int, int]) -> None:  # pragma: no cover
        from pathlib import Path

        import cv2

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)

    def write(self, frame: Frame) -> None:  # pragma: no cover
        if frame.image is not None:
            self._writer.write(frame.image)

    def release(self) -> None:  # pragma: no cover
        self._writer.release()
