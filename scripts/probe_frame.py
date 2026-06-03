"""Grab a single real frame from the live YouTube stream for line placement."""
import os
import cv2
from assbi.video.youtube import stream_url

os.makedirs("data/output", exist_ok=True)
url = stream_url("https://www.youtube.com/watch?v=7uG-gbg0I8Y", max_height=360)
cap = cv2.VideoCapture(url)
print("opened:", cap.isOpened())
ok, frame = cap.read()
print("read:", ok, "shape:", None if frame is None else frame.shape)
if ok:
    cv2.imwrite("data/output/frame_probe.jpg", frame)
    print("saved data/output/frame_probe.jpg")
cap.release()
