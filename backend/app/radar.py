import math

HORIZONTAL_FOV_DEG = 60.0

# Must mirror the radar SVG's own viewBox and polarToXY in App.jsx exactly -
# zone geometry is normalized against that same screen, since zones are drawn
# by dragging directly on the rendered radar.
RADAR_SVG_W = 680.0
RADAR_SVG_H = 520.0
RADAR_SVG_CX = RADAR_SVG_W / 2.0
RADAR_SVG_CY = RADAR_SVG_H / 2.0
RADAR_SVG_RADIUS = min(RADAR_SVG_W, RADAR_SVG_H) / 2.0 - 28.0


def bearing_range_to_unit_xy(bearing_deg, range_u):
    a = math.radians(bearing_deg - 90.0)
    rr = RADAR_SVG_RADIUS * range_u
    x = RADAR_SVG_CX + rr * math.cos(a)
    y = RADAR_SVG_CY + rr * math.sin(a)
    return x / RADAR_SVG_W, y / RADAR_SVG_H


def pixel_to_radar(cx, cy, frame_w, frame_h):
    norm_x = (cx - frame_w / 2.0) / (frame_w / 2.0)  # -1 (left edge) .. +1 (right edge)
    bearing = (norm_x * (HORIZONTAL_FOV_DEG / 2.0)) % 360.0

    range_u = 1.0 - (cy / frame_h)
    range_u = max(0.0, min(1.0, range_u))

    return bearing, range_u
