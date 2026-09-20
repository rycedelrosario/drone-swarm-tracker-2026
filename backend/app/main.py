import asyncio
import math
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.blob_detector import BlobDetector
from app.radar import pixel_to_radar
from app.tracker import MultiObjectTracker

VIDEO_PATH = Path(__file__).resolve().parent.parent / "data" / "perdix_swarm_demo.mp4"
HEADING_SPEED_NORM = 10.0

app = FastAPI(title="Drone Swarm Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127.0.0.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


def alt_band_for(range_u):
    if range_u < 0.35:
        return "LOW"
    if range_u < 0.7:
        return "MED"
    return "HIGH"


def track_to_payload(track, frame_w, frame_h, fps):
    bearing, range_u = pixel_to_radar(track.cx, track.cy, frame_w, frame_h)
    heading = math.degrees(math.atan2(track.vx, -track.vy)) % 360.0
    speed_px_per_sec = math.hypot(track.vx, track.vy) * fps
    rel_speed_u = min(30.0, speed_px_per_sec / HEADING_SPEED_NORM)

    return {
        "id": track.id,
        "callsign": f"UAV-{track.id:02d}",
        "type": "unknown",
        "bearing": round(bearing, 1),
        "range_u": round(range_u, 3),
        "heading": round(heading, 1),
        "rel_speed_u": round(rel_speed_u, 1),
        "alt_band": alt_band_for(range_u),
        "confidence": round(track.confidence, 2),
        "flags": ["LOW_CONF"] if track.confidence < 0.5 else [],
    }


@app.websocket("/ws/tracks")
async def ws_tracks(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    await websocket.send_json({"type": "hello", "frame_w": frame_w, "frame_h": frame_h, "fps": fps})

    detector = BlobDetector()
    tracker = MultiObjectTracker()
    frame_interval = 1.0 / fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            boxes = detector.detect(frame)
            tracks = tracker.update(boxes)
            payload = [track_to_payload(t, frame_w, frame_h, fps) for t in tracks]

            await websocket.send_json({"type": "tracks_snapshot", "tracks": payload})
            await asyncio.sleep(frame_interval)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
