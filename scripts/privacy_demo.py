"""Verify privacy anonymisation on a real frame with real YOLO detections."""
import cv2

from assbi.detection.yolo_detector import YOLODetector
from assbi.domain.interfaces import Frame
from assbi.pipeline.annotator import FrameAnnotator

img = cv2.imread("data/output/frame_probe.jpg")
frame = Frame(index=0, image=img.copy(), width=img.shape[1], height=img.shape[0])

detections = YOLODetector(confidence=0.12).detect(frame)
people = [d for d in detections if d.object_class.value == "person"]
from collections import Counter
print("classes:", Counter(d.object_class.value for d in detections))
print(f"real detections: {len(detections)} ({len(people)} people)")

# Blur only people (default target); vehicles stay visible for analytics.
blurred = img.copy()
FrameAnnotator([], privacy_mode="blur", privacy_strength=15).anonymize(blurred, detections)
cv2.imwrite("data/output/privacy_blur.jpg", blurred)

pixel = img.copy()
FrameAnnotator([], privacy_mode="pixelate", privacy_strength=10).anonymize(pixel, detections)
cv2.imwrite("data/output/privacy_pixelate.jpg", pixel)
print("saved data/output/privacy_blur.jpg and data/output/privacy_pixelate.jpg")
