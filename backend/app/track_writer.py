import threading

from app.db import get_db

INSERT_SQL = """
INSERT INTO track_samples (ts_ms, track_id, bearing, range_u, heading, rel_speed_u, altitude_m, confidence)
VALUES (:ts_ms, :track_id, :bearing, :range_u, :heading, :rel_speed_u, :altitude_m, :confidence)
"""


class TrackSampleWriter:
    def __init__(self, conn=None, flush_interval_s=0.2, flush_row_threshold=500, max_buffer_rows=5000):
        self.conn = conn or get_db()
        self.flush_interval_s = flush_interval_s
        self.flush_row_threshold = flush_row_threshold
        self.max_buffer_rows = max_buffer_rows

        self._buffer = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def add_sample(self, sample):
        with self._lock:
            self._buffer.append(sample)
            if len(self._buffer) > self.max_buffer_rows:
                del self._buffer[: len(self._buffer) - self.max_buffer_rows]
            if len(self._buffer) >= self.flush_row_threshold:
                self._flush_locked()

    def _flush_loop(self):
        while not self._stop.wait(self.flush_interval_s):
            with self._lock:
                self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        rows = self._buffer
        self._buffer = []
        with self.conn:
            self.conn.executemany(INSERT_SQL, rows)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        with self._lock:
            self._flush_locked()
        self.conn.close()
