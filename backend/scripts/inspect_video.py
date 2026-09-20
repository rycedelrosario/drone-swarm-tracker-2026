import cv2

VIDEO_PATH = "data/perdix_swarm_demo.mp4"
OUT_PATH = "data/first_frame.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Resolution: {width}x{height}")
print(f"Total frames: {frame_count}")

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Failed to read first frame")

cv2.imwrite(OUT_PATH, frame)
print(f"Saved first frame to {OUT_PATH}")

cap.release()
