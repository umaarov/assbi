"""Application layer: orchestration and composition."""
from .factory import build_detector, build_lines, build_pipeline, build_video_source
from .orchestrator import PipelineResult, SurveillancePipeline

__all__ = [
    "build_detector",
    "build_lines",
    "build_pipeline",
    "build_video_source",
    "PipelineResult",
    "SurveillancePipeline",
]
