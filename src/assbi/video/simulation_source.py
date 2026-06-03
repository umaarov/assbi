"""A headless video source that yields blank frames for the synthetic demo.

Paired with :class:`~assbi.detection.simulation_detector.SimulationDetector`,
it lets the entire pipeline run without any video file or codec. Frames carry
no pixel data (``image=None``) unless OpenCV is available and ``render=True``,
in which case a simple canvas is produced so an annotated demo video can be
written.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..domain.interfaces import Frame, VideoSource


class SimulationVideoSource(VideoSource):
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 25.0,
        total_frames: int = 500,
        render: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self._fps = fps
        self.total_frames = total_frames
        self.render = render
        self._np = None
        if render:
            try:
                import numpy as np  # noqa: WPS433

                self._np = np
            except ImportError:
                self.render = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.width, self.height

    def frames(self) -> Iterator[Frame]:
        for index in range(self.total_frames):
            image = None
            if self.render and self._np is not None:
                # Asphalt-grey canvas; the orchestrator overlays detections.
                image = self._np.full((self.height, self.width, 3), 60, dtype="uint8")
            yield Frame(index=index, image=image, width=self.width, height=self.height)
