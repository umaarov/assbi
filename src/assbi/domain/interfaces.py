"""Abstract ports (interfaces) that the application layer depends on.

Following the Dependency Inversion Principle, high-level pipeline logic depends
on these abstractions, never on concrete YOLO / OpenCV / SQLite classes. That
keeps the core testable and lets adapters be swapped freely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from .models import CrossingEvent, Detection, FrameAnalytics, SessionSummary


class Frame:
    """Lightweight container for a decoded video frame.

    ``image`` is an opaque object (a numpy array when OpenCV is present, or
    ``None`` in headless/simulation mode). The pipeline never inspects it
    directly — only adapters do.
    """

    __slots__ = ("index", "image", "width", "height")

    def __init__(self, index: int, image, width: int, height: int) -> None:
        self.index = index
        self.image = image
        self.width = width
        self.height = height


class VideoSource(ABC):
    """A source of frames: a file, a camera, or a synthetic generator."""

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) in pixels."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]: ...

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class ObjectDetector(ABC):
    """Detects objects of interest in a single frame."""

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]: ...

    def warmup(self) -> None:  # pragma: no cover - optional
        """Optionally run a dummy inference so the first real frame is fast."""


class AnalyticsRepository(ABC):
    """Persistence port for the analytics warehouse."""

    @abstractmethod
    def start_session(self, session_id: str, source: str) -> None: ...

    @abstractmethod
    def save_frame(self, session_id: str, frame: FrameAnalytics) -> None: ...

    @abstractmethod
    def save_crossing(self, session_id: str, event: CrossingEvent) -> None: ...

    @abstractmethod
    def save_summary(self, summary: SessionSummary) -> None: ...

    @abstractmethod
    def frame_series(self, session_id: str) -> list[FrameAnalytics]: ...

    @abstractmethod
    def crossings(self, session_id: str) -> list[CrossingEvent]: ...

    @abstractmethod
    def summary(self, session_id: str) -> SessionSummary | None: ...

    @abstractmethod
    def list_sessions(self) -> list[SessionSummary]: ...
