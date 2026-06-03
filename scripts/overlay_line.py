"""Draw candidate counting line(s) on the probe frame to verify placement."""
import cv2

frame = cv2.imread("data/output/frame_probe.jpg")

# Candidate: vertical line across the roadway (vehicles travel left<->right).
x = 300
cv2.line(frame, (x, 90), (x, 320), (0, 0, 255), 2)
cv2.putText(frame, "count_line", (x + 6, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite("data/output/frame_line.jpg", frame)
print("saved data/output/frame_line.jpg")
