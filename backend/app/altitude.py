import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "camera_config.json"


def load_camera_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def elevation_angle_deg(pixel_y, horizon_pixel_y, vertical_fov_deg, frame_h):
    deg_per_pixel = vertical_fov_deg / frame_h
    return (horizon_pixel_y - pixel_y) * deg_per_pixel


def solve_landmark_range_m(landmark_elevation_m, camera_height_m, elevation_deg):
    """General case for a non-horizon landmark: solve the ground range implied
    by its known elevation and measured elevation angle. Degenerate at
    elevation_deg == 0 (the horizon), where any range satisfies the equation —
    that landmark type calibrates the pixel->angle zero point instead.
    """
    theta = math.radians(elevation_deg)
    if abs(math.tan(theta)) < 1e-9:
        return None
    return (landmark_elevation_m - camera_height_m) / math.tan(theta)


def altitude_m(pixel_y, range_m, camera_height_m, horizon_pixel_y, vertical_fov_deg, frame_h):
    theta_deg = elevation_angle_deg(pixel_y, horizon_pixel_y, vertical_fov_deg, frame_h)
    return camera_height_m + range_m * math.tan(math.radians(theta_deg))
