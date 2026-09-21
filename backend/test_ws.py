import asyncio
import sqlite3

import websockets

DB_PATH = "data/radar.db"
MAX_RANGE_MS = 10 * 60 * 1000


def get_range():
    conn = sqlite3.connect(DB_PATH)
    start_ms, end_ms = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM track_samples").fetchone()
    conn.close()
    if start_ms is None:
        raise RuntimeError(f"{DB_PATH} has no track_samples yet — run the backend against /ws/tracks first")
    return start_ms, min(end_ms, start_ms + MAX_RANGE_MS - 1000)


async def main():
    start_ms, end_ms = get_range()
    uri = f"ws://127.0.0.1:8000/ws/replay?start_ms={start_ms}&end_ms={end_ms}&rate=1&hz=10"
    async with websockets.connect(uri) as ws:
        for _ in range(10):
            print(await ws.recv())


asyncio.run(main())
