import asyncio
import math
import os
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from app.altitude import altitude_m as compute_altitude_m
from app.altitude import load_camera_config
from app.blob_detector import BlobDetector
from app.db import get_db
from app.radar import pixel_to_radar
from app.track_writer import TrackSampleWriter
from app.tracker import MultiObjectTracker

VIDEO_PATH = Path(__file__).resolve().parent.parent / "data" / "perdix_swarm_demo.mp4"
HEADING_SPEED_NORM = 10.0
RANGE_MAX_M = 5000.0  # rough max visible ground range in this clip's clear desert air
SAMPLE_INTERVAL_S = 0.1  # ~10Hz persistence, decoupled from the video's ~30fps frame rate
REPLAY_MAX_RANGE_MS = 10 * 60 * 1000
REPLAY_MAX_LIMIT = 20000

CAMERA_CONFIG = load_camera_config()
CAMERA_HEIGHT_M = CAMERA_CONFIG["camera_height_m"]
VERTICAL_FOV_DEG = CAMERA_CONFIG["vertical_fov_deg"]
HORIZON_PIXEL_Y = CAMERA_CONFIG["reference_points"][0]["pixel_y"]

sample_writer = TrackSampleWriter()
read_conn = get_db()


def make_detector():
    if os.environ.get("DETECTOR", "blob") == "yolo":
        from app.yolo_detector import YoloDetector
        return YoloDetector()
    return BlobDetector()

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


@app.get("/video")
async def video():
    return FileResponse(VIDEO_PATH, media_type="video/mp4")


@app.get("/replay")
async def replay(
    start_ms: int = Query(...),
    end_ms: int = Query(...),
    limit: int = Query(1000, ge=1, le=REPLAY_MAX_LIMIT),
):
    if start_ms >= end_ms:
        raise HTTPException(400, "start_ms must be less than end_ms")
    if end_ms - start_ms > REPLAY_MAX_RANGE_MS:
        raise HTTPException(400, f"range must not exceed {REPLAY_MAX_RANGE_MS} ms (10 minutes)")

    rows = read_conn.execute(
        "SELECT ts_ms, track_id, bearing, range_u, heading, rel_speed_u, altitude_m, confidence "
        "FROM track_samples WHERE ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms LIMIT ?",
        (start_ms, end_ms, limit),
    ).fetchall()

    columns = ["ts_ms", "track_id", "bearing", "range_u", "heading", "rel_speed_u", "altitude_m", "confidence"]
    return {"rows": [dict(zip(columns, row)) for row in rows]}


def alt_band_for(range_u):
    if range_u < 0.35:
        return "LOW"
    if range_u < 0.7:
        return "MED"
    return "HIGH"


def sample_to_track(track_id, bearing, range_u, heading, rel_speed_u, altitude_m, confidence):
    return {
        "id": track_id,
        "callsign": f"UAV-{track_id:02d}",
        "type": "unknown",
        "bearing": bearing,
        "range_u": range_u,
        "heading": heading,
        "rel_speed_u": rel_speed_u,
        "alt_band": alt_band_for(range_u),
        "altitude_m": altitude_m,
        "confidence": confidence,
        "flags": ["LOW_CONF"] if confidence < 0.5 else [],
    }


def track_to_payload(track, frame_w, frame_h, fps):
    bearing, range_u = pixel_to_radar(track.cx, track.cy, frame_w, frame_h)
    heading = math.degrees(math.atan2(track.vx, -track.vy)) % 360.0
    speed_px_per_sec = math.hypot(track.vx, track.vy) * fps
    rel_speed_u = min(30.0, speed_px_per_sec / HEADING_SPEED_NORM)

    range_m = range_u * RANGE_MAX_M
    alt_m = compute_altitude_m(track.cy, range_m, CAMERA_HEIGHT_M, HORIZON_PIXEL_Y, VERTICAL_FOV_DEG, frame_h)

    return sample_to_track(
        track.id,
        round(bearing, 1),
        round(range_u, 3),
        round(heading, 1),
        round(rel_speed_u, 1),
        round(alt_m),
        round(track.confidence, 2),
    )


@app.websocket("/ws/tracks")
async def ws_tracks(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    await websocket.send_json({"type": "hello", "frame_w": frame_w, "frame_h": frame_h, "fps": fps})

    detector = make_detector()
    tracker = MultiObjectTracker()
    frame_interval = 1.0 / fps
    frame_idx = 0
    last_sample_t = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            boxes = detector.detect(frame)
            tracks = tracker.update(boxes)
            payload = [track_to_payload(t, frame_w, frame_h, fps) for t in tracks]
            bboxes = [[round(c) for c in t.box] for t in tracks]

            now = time.monotonic()
            if now - last_sample_t >= SAMPLE_INTERVAL_S:
                last_sample_t = now
                ts_ms = int(time.time() * 1000)
                for p in payload:
                    sample_writer.add_sample({
                        "ts_ms": ts_ms,
                        "track_id": p["id"],
                        "bearing": p["bearing"],
                        "range_u": p["range_u"],
                        "heading": p["heading"],
                        "rel_speed_u": p["rel_speed_u"],
                        "altitude_m": p["altitude_m"],
                        "confidence": p["confidence"],
                    })

            await websocket.send_json({
                "type": "tracks_snapshot",
                "media_t_sec": round(frame_idx / fps, 3),
                "tracks": payload,
                "bboxes": bboxes,
            })
            frame_idx += 1
            await asyncio.sleep(frame_interval)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


@app.websocket("/ws/replay")
async def ws_replay(
    websocket: WebSocket,
    start_ms: int = Query(...),
    end_ms: int = Query(...),
    rate: float = Query(1.0, gt=0),
    hz: float = Query(10.0, gt=0),
):
    if start_ms >= end_ms:
        await websocket.close(code=1008, reason="start_ms must be less than end_ms")
        return
    if end_ms - start_ms > REPLAY_MAX_RANGE_MS:
        await websocket.close(code=1008, reason="range must not exceed 10 minutes")
        return

    await websocket.accept()

    rows = read_conn.execute(
        "SELECT ts_ms, track_id, bearing, range_u, heading, rel_speed_u, altitude_m, confidence "
        "FROM track_samples WHERE ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms",
        (start_ms, end_ms),
    ).fetchall()

    snapshots = {}
    for row_ts, track_id, bearing, range_u, heading, rel_speed_u, altitude_m, confidence in rows:
        snapshots.setdefault(row_ts, []).append(
            sample_to_track(track_id, bearing, range_u, heading, rel_speed_u, altitude_m, confidence)
        )

    # hz picks which recorded snapshots to send (skip ones closer together than 1000/hz ms of
    # recorded time); rate then scales the wall-clock pacing between those, derived from the
    # actual recorded time gaps rather than a flat interval, so it stays faithful to playback speed.
    min_gap_ms = 1000.0 / hz
    selected_ts = []
    last_emitted = None
    for ts in sorted(snapshots.keys()):
        if last_emitted is None or ts - last_emitted >= min_gap_ms:
            selected_ts.append(ts)
            last_emitted = ts

    try:
        prev_ts = None
        for ts in selected_ts:
            if prev_ts is not None:
                await asyncio.sleep(max(0.0, (ts - prev_ts) / 1000.0 / rate))
            prev_ts = ts
            await websocket.send_json({
                "type": "tracks_snapshot",
                "replay_ts_ms": ts,
                "tracks": snapshots[ts],
            })
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass
