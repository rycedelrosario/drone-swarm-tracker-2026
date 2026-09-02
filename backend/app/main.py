import asyncio
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.vision import (
    CentroidTracker,
    BlobDetector,
    YoloDetector,
    load_altitude_config,
    estimate_altitude
)

app = FastAPI(title="Drone Swarm Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "perdix_demo.mp4")
CONFIG_PATH = os.path.join(BASE_DIR, "data", "altitude_config.json")

alt_config = load_altitude_config(CONFIG_PATH)

# 2. Select detector via DETECTOR environment variable
# Change this line in backend/app/main.py:
DETECTOR_TYPE = os.getenv("DETECTOR", "yolo").lower()
if DETECTOR_TYPE == "yolo":
    print("Initializing YOLOv8 Detector...")
    detector = YoloDetector(conf_thresh=0.12)
else:
    print("Initializing Otsu Blob Detector...")
    detector = BlobDetector()

@app.get("/health")
async def health_check():
    return {
        "ok": True,
        "video_exists": os.path.exists(VIDEO_PATH),
        "detector": DETECTOR_TYPE
    }

@app.get("/video")
async def get_video():
    if os.path.exists(VIDEO_PATH):
        return FileResponse(VIDEO_PATH, media_type="video/mp4")
    return {"error": "Video file not found"}

@app.websocket("/ws/tracks")
async def websocket_tracks(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None
    tracker = CentroidTracker(max_disappeared=30)
    
    # Fast-forward past jet launch intro directly to active drone swarm sequence (~46s)
    START_FRAME = 1400
    # Ensure START_FRAME is within valid range
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap and cap.isOpened()) else 0
    START_FRAME = min(1400, max(0, total_frames - 30)) if total_frames > 0 else 0

    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if (cap and cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0) else 1024
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if (cap and cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0) else 576

    await websocket.send_json({
        "type": "hello",
        "frame_w": frame_w,
        "frame_h": frame_h,
        "fps": 30
    })

    try:
        while True:
            tracks_payload = []

            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
                    ret, frame = cap.read()

                if ret:
                    boxes = detector.detect(frame)
                    tracked_objects = tracker.update(boxes)

                    for obj_id, data in tracked_objects.items():
                        cx, cy = data["cx"], data["cy"]
                        x1, y1, x2, y2 = data["bbox"]
                        heading = data["heading"]
                        rel_speed_u = data["rel_speed_u"]

                        norm_x = (cx - (frame_w / 2)) / (frame_w / 2)
                        norm_y = (cy - (frame_h / 2)) / (frame_h / 2)

                        bearing = (round((cv2.fastAtan2(norm_x, -norm_y)), 1)) % 360.0
                        range_u = min(0.95, max(0.08, round(float(np.hypot(norm_x, norm_y)), 2)))
                        alt_m = estimate_altitude(cy, frame_h, alt_config)

                        # Real normalized bounding box coordinates (0.0 to 1.0)
                        norm_box = [
                            max(0.0, float(x1) / frame_w),
                            max(0.0, float(y1) / frame_h),
                            min(1.0, float(x2) / frame_w),
                            min(1.0, float(y2) / frame_h)
                        ]

                        tracks_payload.append({
                            "id": obj_id,
                            "callsign": f"UAV-{str(obj_id).zfill(2)}",
                            "type": "multirotor" if obj_id % 2 == 0 else "fixedwing",
                            "bearing": bearing,
                            "range_u": range_u,
                            "heading": heading,
                            "rel_speed_u": rel_speed_u,
                            "alt_band": "LOW" if range_u < 0.35 else "MED" if range_u < 0.7 else "HIGH",
                            "altitude_m": alt_m,
                            "confidence": 0.92,
                            "flags": [],
                            "bbox": norm_box
                        })

            await websocket.send_json({
                "type": "tracks_snapshot",
                "tracks": tracks_payload
            })
            await asyncio.sleep(0.066)

    except WebSocketDisconnect:
        if cap:
            cap.release()