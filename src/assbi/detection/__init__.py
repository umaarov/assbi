"""Object detection adapters."""
from .simulation_detector import SimulationDetector

__all__ = ["SimulationDetector"]

# YOLODetector is imported lazily to avoid pulling in ultralytics/torch unless
# the caller actually needs real inference.


def __getattr__(name: str):  # pragma: no cover - thin lazy shim
    if name == "YOLODetector":
        from .yolo_detector import YOLODetector

        return YOLODetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
