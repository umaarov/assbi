"""The pipeline orchestrator — the application layer's use-case entry point.

It wires the configured adapters together and runs the end-to-end flow for one
analysis session:

    video source -> detector -> tracker -> line counter
                                        -> crowd density -> anomaly detector
                 -> per-frame persistence -> (optional annotated video)
                 -> session summary roll-up

It depends only on the domain *ports*, so swapping the simulation backend for
real YOLO + OpenCV changes nothing here.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..analytics.anomaly import RollingAnomalyDetector
from ..analytics.crowd import CrowdAnalyzer
from ..analytics.line_counter import LineCounter
from ..domain.geometry import CountingLine, Point
from ..domain.interfaces import (
    AnalyticsRepository,
    ObjectDetector,
    VideoSource,
)
from ..domain.models import FrameAnalytics, SessionSummary
from ..tracking.centroid_tracker import CentroidTracker
from .annotator import FrameAnnotator, VideoWriter

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    summary: SessionSummary
    line_breakdown: dict


@dataclass
class LiveUpdate:
    """Per-frame snapshot pushed to an ``on_frame`` observer (e.g. a live UI)."""
    frame_index: int
    image: Any                 # annotated BGR frame (numpy array) or None
    people_in: int
    people_out: int
    vehicles_in: int
    vehicles_out: int
    person_count: int
    vehicle_count: int
    density_level: str
    is_anomaly: bool


class SurveillancePipeline:
    def __init__(
        self,
        detector: ObjectDetector,
        tracker: CentroidTracker,
        line_counter: LineCounter,
        crowd: CrowdAnalyzer,
        anomaly: RollingAnomalyDetector,
        repository: AnalyticsRepository,
        lines: list[CountingLine],
        privacy_mode: str = "off",
        privacy_targets: list[str] | None = None,
        privacy_strength: int = 23,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.line_counter = line_counter
        self.crowd = crowd
        self.anomaly = anomaly
        self.repository = repository
        self.lines = lines
        self.privacy_mode = privacy_mode
        self.privacy_targets = privacy_targets
        self.privacy_strength = privacy_strength

    def run(
        self,
        source: VideoSource,
        session_id: str,
        source_label: str,
        *,
        render_path: str | None = None,
        progress_every: int = 50,
        on_frame: Callable[[LiveUpdate], Any] | None = None,
        start_time: datetime | None = None,
    ) -> PipelineResult:
        """Run the analysis loop for one session.

        ``on_frame``, if given, is called once per processed frame with a
        :class:`LiveUpdate` (including the annotated image) so a live UI can
        render the stream in real time. If the callback returns ``False`` the
        run stops early (e.g. the user pressed Stop).

        ``start_time`` anchors footage-relative timestamps: every frame and
        crossing is stamped ``start_time + frame_index / fps`` so the warehouse
        carries the *video's* timeline (not wall-clock processing time). This is
        what lets the dashboard and chatbot answer "which hour was busiest?" or
        "how many crossed in the last 30 minutes?" on a batch-processed file.
        Defaults to now, so a 12-hour clip yields a 12-hour span of timestamps.
        """
        self.repository.start_session(session_id, source_label)
        # fps is needed up-front to map frame indices to footage time.
        fps = source.fps or 25.0
        base_ts = start_time or datetime.now(timezone.utc)
        last_index = 0
        annotator = FrameAnnotator(
            self.lines,
            privacy_mode=self.privacy_mode,
            privacy_targets=self.privacy_targets,
            privacy_strength=self.privacy_strength,
        )
        writer = self._make_writer(render_path, source, annotator)
        # Annotate frames if we're writing a video OR feeding a live observer.
        annotate = (writer is not None or on_frame is not None) and annotator.available

        frames_processed = 0
        peak_crowd = 0
        peak_frame = 0
        anomalies = 0
        confidence_sum = 0.0
        confidence_n = 0
        started = time.perf_counter()

        # A live stream runs until the user stops it (Ctrl+C). Catch that here so
        # the session is still finalised — summary saved — instead of aborting.
        try:
          for frame in source.frames():
            # Footage-relative timestamp for this frame (see ``start_time`` doc).
            frame_ts = base_ts + timedelta(seconds=frame.index / fps)
            last_index = frame.index
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame.index)
            events = self.line_counter.process(tracks, frame.index)
            for event in events:
                # Re-stamp with footage time (the counter uses wall-clock).
                self.repository.save_crossing(session_id, replace(event, timestamp=frame_ts))

            snapshot = self.crowd.snapshot(tracks)
            anomaly = self.anomaly.update(float(snapshot.person_count))
            if anomaly.is_anomaly:
                anomalies += 1

            if snapshot.person_count > peak_crowd:
                peak_crowd = snapshot.person_count
                peak_frame = frame.index

            for d in detections:
                confidence_sum += d.confidence
                confidence_n += 1

            self.repository.save_frame(
                session_id,
                FrameAnalytics(
                    frame_index=frame.index,
                    timestamp=frame_ts,
                    person_count=snapshot.person_count,
                    vehicle_count=snapshot.vehicle_count,
                    total_detections=len(detections),
                    crossings_in=self.line_counter.people_in + self.line_counter.vehicles_in,
                    crossings_out=self.line_counter.people_out + self.line_counter.vehicles_out,
                    is_anomaly=anomaly.is_anomaly,
                    anomaly_score=round(anomaly.score, 3),
                ),
            )

            if annotate:
                annotator.draw(frame, tracks, self.line_counter, snapshot)
            if writer is not None:
                writer.write(frame)

            frames_processed += 1

            if on_frame is not None:
                cont = on_frame(LiveUpdate(
                    frame_index=frame.index,
                    image=frame.image,
                    people_in=self.line_counter.people_in,
                    people_out=self.line_counter.people_out,
                    vehicles_in=self.line_counter.vehicles_in,
                    vehicles_out=self.line_counter.vehicles_out,
                    person_count=snapshot.person_count,
                    vehicle_count=snapshot.vehicle_count,
                    density_level=snapshot.density_level,
                    is_anomaly=anomaly.is_anomaly,
                ))
                if cont is False:
                    break

            if progress_every and frames_processed % progress_every == 0:
                logger.info("processed %d frames (live crowd=%d)",
                            frames_processed, snapshot.person_count)
        except KeyboardInterrupt:
            logger.info("interrupted by user — finalising session %s (%d frames)",
                        session_id, frames_processed)

        if writer is not None:
            writer.release()

        elapsed = time.perf_counter() - started
        # True footage span = position of the last frame read / fps. Using the
        # last index (not the emitted count) keeps duration correct under
        # striding, where we only process every Nth frame of a long video.
        summary = SessionSummary(
            session_id=session_id,
            source=source_label,
            frames_processed=frames_processed,
            duration_seconds=round((last_index + 1) / fps, 2),
            people_in=self.line_counter.people_in,
            people_out=self.line_counter.people_out,
            vehicles_in=self.line_counter.vehicles_in,
            vehicles_out=self.line_counter.vehicles_out,
            peak_crowd=peak_crowd,
            peak_crowd_frame=peak_frame,
            anomalies=anomalies,
            avg_confidence=round(confidence_sum / confidence_n, 3) if confidence_n else 0.0,
        )
        self.repository.save_summary(summary)
        logger.info("session %s done in %.2fs wall, %d frames", session_id, elapsed, frames_processed)
        return PipelineResult(summary=summary, line_breakdown=self.line_counter.breakdown())

    def _make_writer(self, render_path, source, annotator) -> VideoWriter | None:
        if not render_path:
            return None
        if not annotator.available:
            logger.warning("OpenCV not available; skipping annotated video output.")
            return None
        return VideoWriter(render_path, source.fps, source.frame_size)  # pragma: no cover

    @staticmethod
    def lines_from_config(line_configs) -> list[CountingLine]:
        return [
            CountingLine(Point(*lc.start), Point(*lc.end), lc.name)
            for lc in line_configs
        ]
