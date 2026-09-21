import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse

from app.altitude import altitude_m as compute_altitude_m
from app.altitude import load_camera_config
from app.blob_detector import BlobDetector
from app.db import get_db
from app.radar import bearing_range_to_unit_xy, pixel_to_radar
from app.track_writer import TrackSampleWriter
from app.tracker import MultiObjectTracker
from app.zone_events import ZoneEventEngine

VIDEO_PATH = Path(__file__).resolve().parent.parent / "data" / "perdix_swarm_demo.mp4"
HEADING_SPEED_NORM = 10.0
RANGE_MAX_M = 5000.0  # rough max visible ground range in this clip's clear desert air
SAMPLE_INTERVAL_S = 0.1  # ~10Hz persistence, decoupled from the video's ~30fps frame rate
ZONE_REFRESH_INTERVAL_S = 5.0  # how often a connection re-reads /zones for newly drawn ones
REPLAY_MAX_RANGE_MS = 10 * 60 * 1000
REPLAY_MAX_LIMIT = 20000
SQLITE_MAX_INTEGER = 9223372036854775807  # 64-bit signed max; larger values overflow SQLite's INTEGER

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
    start_ms: int = Query(..., ge=0, le=SQLITE_MAX_INTEGER),
    end_ms: int = Query(..., ge=0, le=SQLITE_MAX_INTEGER),
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


@app.get("/events")
async def get_events(
    start_ms: int = Query(..., ge=0, le=SQLITE_MAX_INTEGER),
    end_ms: int = Query(..., ge=0, le=SQLITE_MAX_INTEGER),
    limit: int = Query(1000, ge=1, le=REPLAY_MAX_LIMIT),
):
    if start_ms >= end_ms:
        raise HTTPException(400, "start_ms must be less than end_ms")
    if end_ms - start_ms > REPLAY_MAX_RANGE_MS:
        raise HTTPException(400, f"range must not exceed {REPLAY_MAX_RANGE_MS} ms (10 minutes)")

    rows = read_conn.execute(
        "SELECT events.ts_ms, events.track_id, events.zone_id, events.event_type, zones.name "
        "FROM events LEFT JOIN zones ON zones.id = events.zone_id "
        "WHERE events.ts_ms >= ? AND events.ts_ms <= ? ORDER BY events.ts_ms LIMIT ?",
        (start_ms, end_ms, limit),
    ).fetchall()

    return {
        "events": [
            {
                "ts_ms": ts_ms,
                "track_id": track_id,
                "callsign": f"UAV-{track_id:02d}",
                "zone_id": zone_id,
                "zone_name": zone_name or "?",
                "type": event_type,
            }
            for ts_ms, track_id, zone_id, event_type, zone_name in rows
        ]
    }


MAX_ZONE_NAME_LEN = 128
MAX_POLYGON_VERTICES = 256
CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class ZoneCreate(BaseModel):
    name: str
    kind: str
    geometry: dict


def sanitize_zone_name(name):
    return CONTROL_CHARS_RE.sub("", name).strip()


def _in_unit_range(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0


def validate_geometry(kind, geometry):
    if kind == "rect":
        if set(geometry.keys()) != {"x1", "y1", "x2", "y2"}:
            raise HTTPException(400, "rect geometry must have exactly x1, y1, x2, y2")
        x1, y1, x2, y2 = geometry["x1"], geometry["y1"], geometry["x2"], geometry["y2"]
        if not all(_in_unit_range(v) for v in (x1, y1, x2, y2)):
            raise HTTPException(400, "rect coordinates must be numbers in [0, 1]")
        if x1 >= x2 or y1 >= y2:
            raise HTTPException(400, "rect requires x1 < x2 and y1 < y2")
    elif kind == "polygon":
        points = geometry.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise HTTPException(400, "polygon requires a points list with at least 3 points")
        if len(points) > MAX_POLYGON_VERTICES:
            raise HTTPException(400, f"polygon must have at most {MAX_POLYGON_VERTICES} vertices")
        for p in points:
            if not (isinstance(p, list) and len(p) == 2 and all(_in_unit_range(v) for v in p)):
                raise HTTPException(400, "each polygon point must be [x, y] with x, y in [0, 1]")
    else:
        raise HTTPException(400, "kind must be 'rect' or 'polygon'")


def zone_row_to_dict(row):
    zone_id, name, kind, geometry, created_ms = row
    return {"id": zone_id, "name": name, "kind": kind, "geometry": json.loads(geometry), "created_ms": created_ms}


def load_zones():
    rows = read_conn.execute("SELECT id, name, kind, geometry, created_ms FROM zones ORDER BY id").fetchall()
    return [zone_row_to_dict(row) for row in rows]


def persist_and_enrich_events(events, zones_by_id, tracks_by_id):
    if not events:
        return []
    read_conn.executemany(
        "INSERT INTO events (ts_ms, zone_id, track_id, event_type) VALUES (?, ?, ?, ?)",
        [(e["ts_ms"], e["zone_id"], e["track_id"], e["type"]) for e in events],
    )
    read_conn.commit()
    return [
        {
            "type": e["type"],
            "ts_ms": e["ts_ms"],
            "track_id": e["track_id"],
            "callsign": tracks_by_id.get(e["track_id"], {}).get("callsign", f"UAV-{e['track_id']:02d}"),
            "zone_id": e["zone_id"],
            "zone_name": zones_by_id.get(e["zone_id"], {}).get("name", "?"),
        }
        for e in events
    ]


@app.get("/zones")
async def list_zones():
    rows = read_conn.execute("SELECT id, name, kind, geometry, created_ms FROM zones ORDER BY id").fetchall()
    return {"zones": [zone_row_to_dict(row) for row in rows]}


@app.post("/zones")
async def create_zone(zone: ZoneCreate):
    name = sanitize_zone_name(zone.name)
    if not name:
        raise HTTPException(400, "name is required")
    if len(name) > MAX_ZONE_NAME_LEN:
        raise HTTPException(400, f"name must be at most {MAX_ZONE_NAME_LEN} characters")
    validate_geometry(zone.kind, zone.geometry)

    created_ms = int(time.time() * 1000)
    cur = read_conn.execute(
        "INSERT INTO zones (name, kind, geometry, created_ms) VALUES (?, ?, ?, ?)",
        (name, zone.kind, json.dumps(zone.geometry), created_ms),
    )
    read_conn.commit()

    row = read_conn.execute(
        "SELECT id, name, kind, geometry, created_ms FROM zones WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return zone_row_to_dict(row)


@app.delete("/zones/{zone_id}")
async def delete_zone(zone_id: int = PathParam(..., ge=1, le=SQLITE_MAX_INTEGER)):
    cur = read_conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
    read_conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "zone not found")
    return {"deleted": zone_id}


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
    last_zone_refresh_t = 0.0

    zones = load_zones()
    zone_engine = ZoneEventEngine(zones)

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

                if now - last_zone_refresh_t >= ZONE_REFRESH_INTERVAL_S:
                    last_zone_refresh_t = now
                    zones = load_zones()
                    zone_engine.set_zones(zones)

                zones_by_id = {z["id"]: z for z in zones}
                tracks_by_id = {p["id"]: p for p in payload}
                raw_events = []
                for p in payload:
                    x, y = bearing_range_to_unit_xy(p["bearing"], p["range_u"])
                    raw_events.extend(zone_engine.process(p["id"], x, y, ts_ms))

                zone_events = persist_and_enrich_events(raw_events, zones_by_id, tracks_by_id)
                if zone_events:
                    await websocket.send_json({"type": "events", "events": zone_events})

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
    start_ms: int = Query(..., ge=0, le=SQLITE_MAX_INTEGER),
    end_ms: int = Query(..., ge=0, le=SQLITE_MAX_INTEGER),
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
