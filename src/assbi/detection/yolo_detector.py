"""Ultralytics YOLO adapter — the production object detector.

Imports of ``ultralytics`` / ``cv2`` are deferred to construction time so the
rest of the platform imports cleanly on machines without the heavy ML stack
(e.g. the unit-test runner or the synthetic demo). Install the optional extras
with ``pip install -r requirements.txt`` to enable it.
"""
from __future__ import annotations

from ..domain.geometry import BoundingBox
from ..domain.interfaces import Frame, ObjectDetector
from ..domain.models import Detection, ObjectClass

# COCO class ids YOLO emits that we care about, mapped to our domain enum.
_COCO_TO_CLASS = {
    0: ObjectClass.PERSON,
    1: ObjectClass.BICYCLE,
    2: ObjectClass.CAR,
    3: ObjectClass.MOTORCYCLE,
    5: ObjectClass.BUS,
    7: ObjectClass.TRUCK,
}


class YOLODetector(ObjectDetector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        device: str | None = None,
        classes: list[ObjectClass] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # noqa: WPS433 (deferred heavy import)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install -r requirements.txt` "
                "or use the SimulationDetector for a dependency-free demo."
            ) from exc

        self._model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        wanted = classes or list(_COCO_TO_CLASS.values())
        self._wanted = set(wanted)
        # Restrict YOLO to the COCO ids we map, for speed.
        self._coco_filter = [cid for cid, cls in _COCO_TO_CLASS.items() if cls in self._wanted]

    def warmup(self) -> None:  # pragma: no cover - requires model + hardware
        import numpy as np

        self._model.predict(np.zeros((640, 640, 3), dtype="uint8"), verbose=False)

    def detect(self, frame: Frame) -> list[Detection]:  # pragma: no cover - needs cv2
        results = self._model.predict(
            frame.image,
            conf=self.confidence,
            iou=self.iou,
            classes=self._coco_filter or None,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls.item())
                obj_class = _COCO_TO_CLASS.get(cls_id)
                if obj_class is None or obj_class not in self._wanted:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        object_class=obj_class,
                        box=BoundingBox(x1, y1, x2, y2),
                        confidence=float(box.conf.item()),
                    )
                )
        return detections
