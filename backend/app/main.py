import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Drone Swarm Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"ok": True}

@app.websocket("/ws/tracks")
async def websocket_tracks(websocket: WebSocket):
    await websocket.accept()
    
    await websocket.send_json({
        "type": "hello",
        "frame_w": 1024,
        "frame_h": 576,
        "fps": 30
    })
    
    bearing = 0.0
    try:
        while True:
            bearing = (bearing + 2.0) % 360.0
            heading = (bearing + 90.0) % 360.0
            
            track_payload = {
                "type": "tracks_snapshot",
                "tracks": [
                    {
                        "id": 1,
                        "callsign": "UAV-01",
                        "type": "multirotor",
                        "bearing": round(bearing, 1),
                        "range_u": 0.5,
                        "heading": round(heading, 1),
                        "rel_speed_u": 12.0,
                        "alt_band": "MED",
                        "confidence": 0.95,
                        "flags": []
                    }
                ]
            }
            await websocket.send_json(track_payload)
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        pass
