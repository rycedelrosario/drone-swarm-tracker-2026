import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.zone_events import ZoneEventEngine

ZONE = {"id": 1, "kind": "rect", "geometry": {"x1": 0.2, "y1": 0.2, "x2": 0.6, "y2": 0.6}}
OUTSIDE = (0.9, 0.9)
INSIDE = (0.4, 0.4)
TICK_MS = 200

engine = ZoneEventEngine([ZONE])
all_events = []
ts = 0

for _ in range(5):  # 1s outside before entering
    all_events += engine.process(1, *OUTSIDE, ts)
    ts += TICK_MS

for _ in range(6000 // TICK_MS):  # enter, stay inside 6s
    all_events += engine.process(1, *INSIDE, ts)
    ts += TICK_MS

for _ in range(5):  # leave, stay outside 1s
    all_events += engine.process(1, *OUTSIDE, ts)
    ts += TICK_MS

print("all emitted events:")
for e in all_events:
    print(f"  {e['ts_ms']:>6}ms  {e['type']:<6}  track={e['track_id']} zone={e['zone_id']}")

types = [e["type"] for e in all_events]
print("\nsequence:", types)

expected = ["enter", "dwell", "exit"]
assert types == expected, f"expected {expected}, got {types}"
assert types.count("dwell") == 1, "dwell must fire exactly once"
print("\nPASS: exactly enter, dwell, exit in order, dwell fired once")
