"""OpenCV-backed video source for files, RTSP streams and webcams."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from ..domain.interfaces import Frame, VideoSource

# ffmpeg options handed to OpenCV for network streams so a live feed survives
# transient drops / the CDN rotating segments instead of ending the run. Applied
# only to http(s)/rtsp sources (see ``_is_stream``).
_FFMPEG_STREAM_OPTS = (
    "reconnect;1|reconnect_streamed;1|reconnect_at_eof;1"
    "|reconnect_delay_max;5|rw_timeout;15000000"
)


def _is_stream(source: object) -> bool:
    return isinstance(source, str) and source.startswith(("http", "rtsp", "rtmp"))


class OpenCVVideoSource(VideoSource):
    """Reads frames from any source OpenCV's VideoCapture understands.

    Args:
        source: a file path, an integer camera index, or a stream URL.
        stride: process every Nth frame (1 = every frame). Higher values trade
            temporal resolution for throughput on long videos.
        max_frames: optional hard cap on frames yielded (useful for demos).
        reopen: optional callable returning a *fresh* source (e.g. a re-resolved
            live URL). When a live stream ends or its URL expires, this is called
            to reconnect so the feed keeps running endlessly. Omit for files.
        max_reconnects: how many consecutive reconnect attempts to make before
            giving up (resets after any successful frame).
    """

    def __init__(
        self,
        source: str | int,
        stride: int = 1,
        max_frames: int | None = None,
        reopen: Callable[[], str | int] | None = None,
        max_reconnects: int = 5,
    ) -> None:
        try:
            import cv2  # noqa: WPS433 (deferred heavy import)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "opencv-python is not installed. Run `pip install -r requirements-full.txt`."
            ) from exc

        self._cv2 = cv2
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, str) and not source.startswith(("rtsp", "rtmp", "http")):
            if not Path(source).exists():
                raise FileNotFoundError(f"Video file not found: {source}")
        self._reopen = reopen
        self._max_reconnects = max(0, max_reconnects)
        self._cap = self._open(source)
        self.stride = max(1, stride)
        self.max_frames = max_frames

    def _open(self, source: str | int):
        """Open a VideoCapture, enabling ffmpeg auto-reconnect for streams."""
        import os

        if _is_stream(source):
            # OpenCV reads this env var when constructing an FFMPEG capture; it
            # makes the underlying ffmpeg reconnect on EOF / network hiccups.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_STREAM_OPTS
            cap = self._cv2.VideoCapture(source, self._cv2.CAP_FFMPEG)
        else:
            cap = self._cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        return cap

    @property
    def fps(self) -> float:
        fps = self._cap.get(self._cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def _reconnect(self) -> bool:
        """Try to re-resolve and reopen a live source. Returns True on success."""
        if self._reopen is None:
            return False
        try:
            self._cap.release()
        except Exception:
            pass
        try:
            new_source = self._reopen()
            self._cap = self._open(new_source)
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  ! reconnect failed: {exc}")
            return False

    def frames(self) -> Iterator[Frame]:  # pragma: no cover - needs real video
        index = 0
        emitted = 0
        reconnects = 0
        w, h = self.frame_size
        while True:
            ok, image = self._cap.read()
            if not ok:
                # End of stream. For a live feed, try to reconnect instead of
                # stopping, so monitoring continues endlessly.
                if reconnects < self._max_reconnects and self._reopen is not None:
                    reconnects += 1
                    print(f"  … live stream dropped; reconnecting "
                          f"({reconnects}/{self._max_reconnects}) …")
                    if self._reconnect():
                        w, h = self.frame_size or (w, h)
                        continue
                break
            reconnects = 0  # a good read resets the consecutive-failure counter
            if index % self.stride == 0:
                yield Frame(index=index, image=image, width=w, height=h)
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    break
            index += 1

    def close(self) -> None:
        self._cap.release()
