import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

VIDEO_PATH = "data/perdix_swarm_demo.mp4"
START_FRAME = 3000
NUM_FRAMES = 120  # ~4s at 29.97fps


def run(detector_name):
    os.environ["DETECTOR"] = detector_name
    from app.main import make_detector
    from app.tracker import MultiObjectTracker

    detector = make_detector()
    tracker = MultiObjectTracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    lifespans = {}
    frames_processed = 0
    t0 = time.time()

    for _ in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.detect(frame)
        tracks = tracker.update(boxes)
        for t in tracks:
            lifespans[t.id] = lifespans.get(t.id, 0) + 1
        frames_processed += 1

    elapsed = time.time() - t0
    cap.release()

    tracks_created = len(lifespans)
    avg_len = sum(lifespans.values()) / tracks_created if tracks_created else 0.0
    approx_fps = frames_processed / elapsed if elapsed > 0 else 0.0

    return {
        "detector": detector_name,
        "frames_processed": frames_processed,
        "tracks_created": tracks_created,
        "avg_track_length_frames": round(avg_len, 1),
        "elapsed_sec": round(elapsed, 2),
        "approx_fps": round(approx_fps, 2),
    }


def main():
    results = [run(name) for name in ("blob", "yolo")]
    for r in results:
        print(
            f"{r['detector']:5s}  "
            f"tracks_created={r['tracks_created']:3d}  "
            f"avg_track_length_frames={r['avg_track_length_frames']:5.1f}  "
            f"frames={r['frames_processed']:3d}  "
            f"elapsed={r['elapsed_sec']:6.2f}s  "
            f"approx_fps={r['approx_fps']:5.2f}"
        )


if __name__ == "__main__":
    main()
