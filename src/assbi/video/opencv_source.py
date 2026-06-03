"""OpenCV-backed video source for files, RTSP streams and webcams."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..domain.interfaces import Frame, VideoSource


class OpenCVVideoSource(VideoSource):
    """Reads frames from any source OpenCV's VideoCapture understands.

    Args:
        source: a file path, an integer camera index, or a stream URL.
        stride: process every Nth frame (1 = every frame). Higher values trade
            temporal resolution for throughput on long videos.
        max_frames: optional hard cap on frames yielded (useful for demos).
    """

    def __init__(
        self,
        source: str | int,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        try:
            import cv2  # noqa: WPS433 (deferred heavy import)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "opencv-python is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._cv2 = cv2
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, str) and not source.startswith(("rtsp", "http")):
            if not Path(source).exists():
                raise FileNotFoundError(f"Video file not found: {source}")
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.stride = max(1, stride)
        self.max_frames = max_frames

    @property
    def fps(self) -> float:
        fps = self._cap.get(self._cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def frames(self) -> Iterator[Frame]:  # pragma: no cover - needs real video
        index = 0
        emitted = 0
        w, h = self.frame_size
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if index % self.stride == 0:
                yield Frame(index=index, image=image, width=w, height=h)
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    break
            index += 1

    def close(self) -> None:
        self._cap.release()
