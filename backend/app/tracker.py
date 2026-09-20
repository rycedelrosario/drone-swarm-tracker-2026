from dataclasses import dataclass
from typing import List, Tuple

Box = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(box):
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    id: int
    box: Tuple[float, float, float, float]
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    misses: int = 0
    confidence: float = 0.0


class MultiObjectTracker:
    def __init__(self, iou_thresh=0.1, dist_thresh=80.0, max_misses=20, ema_alpha=0.5):
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.max_misses = max_misses
        self.ema_alpha = ema_alpha
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Box]) -> List[Track]:
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        matches = []

        # Phase 1: IoU matching, best score first.
        pairs = []
        for ti in unmatched_tracks:
            for di in unmatched_dets:
                score = _iou(self.tracks[ti].box, detections[di])
                if score >= self.iou_thresh:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        unmatched_tracks -= used_t
        unmatched_dets -= used_d

        # Phase 2: centroid-distance fallback for whatever IoU couldn't match.
        pairs = []
        for ti in unmatched_tracks:
            tcx, tcy = self.tracks[ti].cx, self.tracks[ti].cy
            for di in unmatched_dets:
                dcx, dcy = _centroid(detections[di])
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                if dist <= self.dist_thresh:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        unmatched_tracks -= used_t
        unmatched_dets -= used_d

        # Apply matches: update center, EMA velocity, confidence; reset misses.
        for ti, di in matches:
            track = self.tracks[ti]
            box = detections[di]
            cx, cy = _centroid(box)
            dx, dy = cx - track.cx, cy - track.cy
            track.vx = self.ema_alpha * dx + (1 - self.ema_alpha) * track.vx
            track.vy = self.ema_alpha * dy + (1 - self.ema_alpha) * track.vy
            track.cx, track.cy = cx, cy
            track.box = box[:4]
            track.confidence = box[4]
            track.age += 1
            track.misses = 0

        # Tracks nothing matched this frame: age but count a miss.
        for ti in unmatched_tracks:
            self.tracks[ti].age += 1
            self.tracks[ti].misses += 1

        # Drop tracks that have been missing too long.
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Start a new track for every detection nothing claimed.
        for di in unmatched_dets:
            box = detections[di]
            cx, cy = _centroid(box)
            self.tracks.append(Track(
                id=self._next_id,
                box=box[:4],
                cx=cx,
                cy=cy,
                confidence=box[4],
            ))
            self._next_id += 1

        return self.tracks
