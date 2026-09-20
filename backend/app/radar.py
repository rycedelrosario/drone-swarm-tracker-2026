HORIZONTAL_FOV_DEG = 60.0


def pixel_to_radar(cx, cy, frame_w, frame_h):
    norm_x = (cx - frame_w / 2.0) / (frame_w / 2.0)  # -1 (left edge) .. +1 (right edge)
    bearing = (norm_x * (HORIZONTAL_FOV_DEG / 2.0)) % 360.0

    range_u = 1.0 - (cy / frame_h)
    range_u = max(0.0, min(1.0, range_u))

    return bearing, range_u
