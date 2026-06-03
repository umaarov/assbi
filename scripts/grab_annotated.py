"""Pull a frame from the annotated output video to verify live rendering."""
import cv2

cap = cv2.VideoCapture("data/output/annotated.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, min(2200, max(0, total - 1)))
ok, frame = cap.read()
if ok:
    cv2.imwrite("data/output/annotated_live_frame.jpg", frame)
    print("saved data/output/annotated_live_frame.jpg", frame.shape, "of", total)
cap.release()
