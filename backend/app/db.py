import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "radar.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS track_samples (
    ts_ms INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    bearing REAL NOT NULL,
    range_u REAL NOT NULL,
    heading REAL NOT NULL,
    rel_speed_u REAL NOT NULL,
    altitude_m REAL NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_track_samples_ts_ms ON track_samples (ts_ms);

CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('rect', 'polygon')),
    geometry TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    track_id INTEGER NOT NULL,
    event_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts_ms ON events (ts_ms);
CREATE INDEX IF NOT EXISTS idx_events_zone_id ON events (zone_id);
"""


def get_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
