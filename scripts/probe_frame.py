"""Grab a real frame from the live stream and draw the configured counting
line(s) on it, so you can verify/adjust line placement for a new camera.

Run:  python scripts/probe_frame.py
Outputs:
  data/output/frame_probe.jpg  — the raw frame
  data/output/frame_line.jpg   — the same frame with the counting line(s) drawn

Open frame_line.jpg and tweak `lines:` in config/config.yaml (authored against
config.video width/height) until the red line sits across the road / footpath.
"""
import os

import cv2

from assbi.config import AppConfig
from assbi.video.youtube import DEFAULT_URL, stream_url

os.makedirs("data/output", exist_ok=True)
config = AppConfig.load("config/config.yaml")

url = stream_url(
    DEFAULT_URL,
    max_height=config.video.height,
    cookies_from_browser=config.youtube.cookies_from_browser,
    cookies_file=config.youtube.cookies_file,
    remote_components=config.youtube.remote_components,
)
cap = cv2.VideoCapture(url)
print("opened:", cap.isOpened())
ok, frame = cap.read()
print("read:", ok, "shape:", None if frame is None else frame.shape)

if ok:
    cv2.imwrite("data/output/frame_probe.jpg", frame)
    h, w = frame.shape[:2]
    rw, rh = config.video.width, config.video.height
    sx, sy = w / rw, h / rh  # config lines are authored at rw x rh; scale to frame
    for lc in config.lines:
        p1 = (int(lc.start[0] * sx), int(lc.start[1] * sy))
        p2 = (int(lc.end[0] * sx), int(lc.end[1] * sy))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.putText(frame, lc.name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite("data/output/frame_line.jpg", frame)
    print(f"saved frame_probe.jpg + frame_line.jpg ({w}x{h}); "
          f"line authored at {rw}x{rh}, scaled x{sx:.2f}/y{sy:.2f}")
cap.release()
