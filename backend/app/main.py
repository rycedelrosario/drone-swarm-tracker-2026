import asyncio
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.vision import CentroidTracker, detect_blobs_otsu

app = FastAPI(title="Drone Swarm Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/perdix_demo.mp4"))

@app.get("/health")
async def health_check():
    return {"ok": True, "video_exists": os.path.exists(VIDEO_PATH)}

@app.websocket("/ws/tracks")
async def websocket_tracks(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None
    tracker = CentroidTracker(max_disappeared=10)
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap and cap.isOpened() else 1024
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap and cap.isOpened() else 576

    await websocket.send_json({
        "type": "hello",
        "frame_w": frame_w,
        "frame_h": frame_h,
        "fps": 30
    })

    synthetic_angle = 0.0

    try:
        while True:
            tracks_payload = []

            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if ret:
                    centroids = detect_blobs_otsu(frame)
                    tracked_objects = tracker.update(centroids)

                    for obj_id, (cx, cy) in tracked_objects.items():
                        norm_x = (cx - (frame_w / 2)) / (frame_w / 2)
                        norm_y = (cy - (frame_h / 2)) / (frame_h / 2)

                        bearing = (round((cv2.fastAtan2(norm_x, -norm_y)), 1)) % 360.0
                        range_u = min(0.95, max(0.08, round(float(np.hypot(norm_x, norm_y)), 2)))

                        tracks_payload.append({
                            "id": obj_id,
                            "callsign": f"UAV-{str(obj_id).zfill(2)}",
                            "type": "multirotor" if obj_id % 2 == 0 else "fixedwing",
                            "bearing": bearing,
                            "range_u": range_u,
                            "heading": (bearing + 45.0) % 360.0,
                            "rel_speed_u": 14.0,
                            "alt_band": "LOW" if range_u < 0.35 else "MED" if range_u < 0.7 else "HIGH",
                            "confidence": 0.92,
                            "flags": []
                        })
            
            if not cap or not cap.isOpened() or len(tracks_payload) == 0:
                synthetic_angle = (synthetic_angle + 3.0) % 360.0
                tracks_payload = [
                    {
                        "id": 101,
                        "callsign": "UAV-01",
                        "type": "multirotor",
                        "bearing": round(synthetic_angle, 1),
                        "range_u": 0.45,
                        "heading": round((synthetic_angle + 90.0) % 360.0, 1),
                        "rel_speed_u": 15.0,
                        "alt_band": "MED",
                        "confidence": 0.96,
                        "flags": []
                    }
                ]

            await websocket.send_json({
                "type": "tracks_snapshot",
                "tracks": tracks_payload
            })
            await asyncio.sleep(0.066)

    except WebSocketDisconnect:
        if cap:
            cap.release()