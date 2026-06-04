"""Composition root: builds a wired pipeline from an :class:`AppConfig`.

Keeping all object construction here (the only place that knows about concrete
adapters) is the Dependency-Injection seam that keeps every other module
testable and backend-agnostic.
"""
from __future__ import annotations

from ..analytics.anomaly import RollingAnomalyDetector
from ..analytics.crowd import CrowdAnalyzer
from ..analytics.line_counter import LineCounter
from ..config import AppConfig
from ..domain.geometry import CountingLine, Point
from ..domain.interfaces import ObjectDetector, VideoSource
from ..persistence.sqlite_repository import SQLiteAnalyticsRepository
from ..tracking.centroid_tracker import CentroidTracker
from .orchestrator import SurveillancePipeline


def build_lines(config: AppConfig) -> list[CountingLine]:
    return [
        CountingLine(Point(*lc.start), Point(*lc.end), lc.name)
        for lc in config.lines
    ]


def build_detector(config: AppConfig) -> ObjectDetector:
    backend = config.detection.backend.lower()
    if backend == "yolo":
        from ..detection.yolo_detector import YOLODetector
        from ..domain.models import ObjectClass

        wanted = [ObjectClass.from_label(c) for c in config.detection.classes]
        wanted = [c for c in wanted if c is not None] or None  # None = all classes
        return YOLODetector(
            model_path=config.detection.model_path,
            confidence=config.detection.confidence,
            iou=config.detection.iou,
            device=config.detection.device,
            classes=wanted,
            imgsz=config.detection.imgsz,
        )
    if backend == "simulation":
        from ..detection.simulation_detector import SimulationDetector

        return SimulationDetector(width=config.video.width, height=config.video.height)
    raise ValueError(f"Unknown detection backend: {config.detection.backend!r}")


def build_video_source(config: AppConfig, source: str | int | None) -> VideoSource:
    """Pick a video source.

    ``source`` None -> synthetic. A YouTube watch URL is resolved to a live
    media URL (streamed, never downloaded). Anything else (file path, RTSP/HTTP
    URL, camera index) is opened directly by OpenCV.
    """
    if source is None:
        from ..video.simulation_source import SimulationVideoSource

        return SimulationVideoSource(
            width=config.video.width,
            height=config.video.height,
            fps=config.video.fps,
            total_frames=config.video.total_frames,
            render=config.video.render,
        )

    from ..video.youtube import is_youtube_url, stream_url

    if is_youtube_url(source):
        # Resolve to a direct CDN URL and stream it — no file on disk.
        source = stream_url(
            source,
            max_height=config.video.height,
            cookies_from_browser=config.youtube.cookies_from_browser,
            cookies_file=config.youtube.cookies_file,
            remote_components=config.youtube.remote_components,
        )

    from ..video.opencv_source import OpenCVVideoSource

    return OpenCVVideoSource(
        source=source,
        stride=config.video.stride,
        max_frames=config.video.max_frames,
    )


def scale_lines_to_source(config: AppConfig, source) -> None:
    """Scale config.lines from the reference resolution to the real frame size.

    Counting lines are authored against ``config.video`` (e.g. 640x360); this
    rescales them to the actual video (e.g. a 1280x720 download or a webcam) so
    they land on the road at any resolution. Without it, crossings count 0 on a
    mismatched-resolution video. Call after building the source, before the
    pipeline.
    """
    try:
        aw, ah = source.frame_size
    except Exception:
        return
    rw, rh = config.video.width, config.video.height
    if not (aw and ah and rw and rh) or (aw == rw and ah == rh):
        return
    sx, sy = aw / rw, ah / rh
    for lc in config.lines:
        lc.start = (lc.start[0] * sx, lc.start[1] * sy)
        lc.end = (lc.end[0] * sx, lc.end[1] * sy)


def build_assistant(config: AppConfig, repository, session_id: str):
    """Build the analytics chatbot, attaching a real LLM backend if a key exists.

    Returns the :class:`SurveillanceAssistant`. When the configured provider's
    API key (e.g. ``$DEEPSEEK_API_KEY``) is present, the assistant answers with a
    grounded LLM; otherwise it falls back to the deterministic rule engine.
    """
    import os

    from ..chatbot.assistant import SurveillanceAssistant

    llm = None
    cc = config.chatbot
    if cc.provider and cc.provider.lower() not in ("none", ""):
        if os.environ.get(cc.api_key_env):
            from ..chatbot.llm import DeepSeekBackend

            llm = DeepSeekBackend(
                model=cc.model,
                base_url=cc.base_url,
                temperature=cc.temperature,
                env_var=cc.api_key_env,
            )
    return SurveillanceAssistant(repository, session_id, llm=llm)


def build_pipeline(config: AppConfig, repository: SQLiteAnalyticsRepository | None = None) -> SurveillancePipeline:
    lines = build_lines(config)
    repo = repository or SQLiteAnalyticsRepository(config.database_path)
    return SurveillancePipeline(
        detector=build_detector(config),
        tracker=CentroidTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_missed=config.tracking.max_missed,
            max_distance=config.tracking.max_distance,
        ),
        line_counter=LineCounter(lines),
        crowd=CrowdAnalyzer(
            moderate=config.crowd.moderate,
            high=config.crowd.high,
            critical=config.crowd.critical,
        ),
        anomaly=RollingAnomalyDetector(
            window=config.anomaly.window,
            threshold=config.anomaly.threshold,
            warmup=config.anomaly.warmup,
            min_scale=config.anomaly.min_scale,
        ),
        repository=repo,
        lines=lines,
        privacy_mode=config.privacy.mode,
        privacy_targets=config.privacy.targets,
        privacy_strength=config.privacy.strength,
    )
