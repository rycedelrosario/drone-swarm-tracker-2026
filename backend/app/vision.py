import cv2
import numpy as np
import math
import json
import os

def load_altitude_config(config_path):
    if not os.path.exists(config_path):
        return {"camera": {"height_m": 1200.0, "vfov_deg": 40.0}, "landmark": {"pixel_y": 360}}
    with open(config_path, "r") as f:   
        return json.load(f)

def estimate_altitude(cy, frame_h, config):
    cam_height = config["camera"]["height_m"]
    vfov = config["camera"]["vfov_deg"]
    
    # Map vertical pixel offset to elevation angle relative to center frame
    norm_y = (cy - (frame_h / 2.0)) / (frame_h / 2.0)
    angle_rad = math.radians(-norm_y * (vfov / 2.0))
    
    # Estimate altitude in meters based on pitch angle and base camera height
    estimated_alt = cam_height + (angle_rad * 450.0)
    return round(max(50.0, min(5000.0, estimated_alt)), 1)

class BlobDetector:
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thresh = cv2.bitwise_or(thresh_inv, thresh_norm)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 16 <= area <= 80000:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, x + w, y + h, 0.85))
        return boxes[:80]


class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.12):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_thresh)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes


class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 1
        self.objects = {}
        self.velocities = {}  # Store (vx, vy) per object ID
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.velocities[self.next_object_id] = (0.0, 0.0)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.velocities[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_results()

        input_centroids = np.zeros((len(centroids), 2), dtype="int")
        for (i, (x1, y1, x2, y2, _)) in enumerate(centroids):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 100:
                    continue

                object_id = object_ids[row]
                
                # Calculate velocity delta (vx, vy) using exponential moving average
                prev_x, prev_y = self.objects[object_id]
                new_x, new_y = input_centroids[col]
                vx = 0.7 * self.velocities[object_id][0] + 0.3 * (new_x - prev_x)
                vy = 0.7 * self.velocities[object_id][1] + 0.3 * (new_y - prev_y)
                
                self.velocities[object_id] = (vx, vy)
                self.objects[object_id] = (new_x, new_y)
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.get_results()

    def get_results(self):
        results = {}
        for obj_id, (cx, cy) in self.objects.items():
            vx, vy = self.velocities.get(obj_id, (0.0, 0.0))
            
            # Heading: degrees(atan2(vx, -vy)) mapped to 0-360
            heading = (math.degrees(math.atan2(vx, -vy)) + 360) % 360
            
            # rel_speed_u: scaled velocity vector magnitude
            speed_norm = 10.0  # NORM scale factor[cite: 2]
            rel_speed_u = min(30.0, math.sqrt(vx**2 + vy**2) * speed_norm)
            
            results[obj_id] = {
                "cx": cx,
                "cy": cy,
                "heading": round(heading, 1),
                "rel_speed_u": round(rel_speed_u, 1)
            }
        return results