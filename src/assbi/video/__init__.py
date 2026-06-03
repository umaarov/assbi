"""Video source adapters."""
from .simulation_source import SimulationVideoSource

__all__ = ["SimulationVideoSource"]


def __getattr__(name: str):  # pragma: no cover - lazy shim for heavy deps
    if name == "OpenCVVideoSource":
        from .opencv_source import OpenCVVideoSource

        return OpenCVVideoSource
    if name == "download_video":
        from .youtube import download_video

        return download_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
