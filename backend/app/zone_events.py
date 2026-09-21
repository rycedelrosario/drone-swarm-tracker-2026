from app.geometry import point_in_polygon, point_in_rect

DWELL_SECONDS = 5.0


class ZoneEventEngine:
    def __init__(self, zones):
        self.zones = zones
        self.state = {}  # (track_id, zone_id) -> {"status", "enter_ts", "dwell_emitted"}

    def set_zones(self, zones):
        self.zones = zones

    def process(self, track_id, x, y, ts_ms):
        events = []
        for zone in self.zones:
            key = (track_id, zone["id"])
            st = self.state.setdefault(key, {"status": "outside", "enter_ts": None, "dwell_emitted": False})
            inside = self._point_in_zone(x, y, zone)

            if inside and st["status"] == "outside":
                st["status"] = "inside"
                st["enter_ts"] = ts_ms
                st["dwell_emitted"] = False
                events.append({"type": "enter", "track_id": track_id, "zone_id": zone["id"], "ts_ms": ts_ms})

            elif not inside and st["status"] == "inside":
                st["status"] = "outside"
                st["enter_ts"] = None
                st["dwell_emitted"] = False
                events.append({"type": "exit", "track_id": track_id, "zone_id": zone["id"], "ts_ms": ts_ms})

            elif inside and st["status"] == "inside" and not st["dwell_emitted"]:
                if (ts_ms - st["enter_ts"]) / 1000.0 >= DWELL_SECONDS:
                    st["dwell_emitted"] = True
                    events.append({"type": "dwell", "track_id": track_id, "zone_id": zone["id"], "ts_ms": ts_ms})

        return events

    def process_batch(self, ts_ms, tracks):
        events = []
        for t in tracks:
            events.extend(self.process(t["id"], t["x"], t["y"], ts_ms))
        return events

    def _point_in_zone(self, x, y, zone):
        geometry = zone["geometry"]
        if zone["kind"] == "rect":
            return point_in_rect(x, y, geometry["x1"], geometry["y1"], geometry["x2"], geometry["y2"])
        if zone["kind"] == "polygon":
            return point_in_polygon(x, y, geometry["points"])
        return False
