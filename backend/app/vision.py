import os
import json
import math
import cv2
import numpy as np

class BlobDetector:
    def detect(self, frame):
        h_f, w_f, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur tuned for small specks
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding picks up tiny faint objects when zoomed out
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # Mask out UI overlay edges
        mask = np.zeros_like(thresh)
        mask[int(h_f * 0.08):int(h_f * 0.88), int(w_f * 0.05):int(w_f * 0.95)] = 255
        thresh = cv2.bitwise_and(thresh, mask)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            # Accept tiny 6px specks (zoomed out) up to large 25000px shapes (zoomed in)
            if 6 <= area <= 25000:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / float(h)
                if 0.2 <= aspect_ratio <= 5.0:
                    boxes.append((x, y, x + w, y + h, 0.85))
        
        # Sort by size to prioritize primary targets
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[:10]


class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.05):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.conf_thresh)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # COCO classes: 0=person, 4=airplane, 14=bird, etc.
            # Accept airplanes or unclassified small object detections
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        # Sort by confidence and return top detections
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        return boxes[:2]

class CentroidTracker:
    def __init__(self, max_disappeared=45):
        self.next_object_id = 1
        self.objects = {}
        self.boxes = {}
        self.velocities = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        # Fill lowest available track slot ID
        assigned_id = 1 if 1 not in self.objects else 2 if 2 not in self.objects else self.next_object_id
        
        self.objects[assigned_id] = centroid
        self.boxes[assigned_id] = box[:4]
        self.velocities[assigned_id] = (0.0, 0.0)
        self.disappeared[assigned_id] = 0
        if assigned_id >= self.next_object_id:
            self.next_object_id = assigned_id + 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.boxes:
            del self.boxes[object_id]
        if object_id in self.velocities:
            del self.velocities[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_results()

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, (x1, y1, x2, y2, _)) in enumerate(detections):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            for i in range(0, min(2, len(input_centroids))):
                self.register(input_centroids[i], detections[i])
        else:
            object_ids = list(self.objects.keys())
            
            # Predict position using current velocity vector to compensate for camera pan/zoom
            predicted_centroids = []
            for obj_id in object_ids:
                cx, cy = self.objects[obj_id]
                vx, vy = self.velocities[obj_id]
                predicted_centroids.append((cx + vx * 1.2, cy + vy * 1.2))

            D = np.linalg.norm(np.array(predicted_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                # Extended search distance (400px) keeps lock during zoom/pan
                if D[row, col] > 400:
                    continue

                object_id = object_ids[row]
                prev_x, prev_y = self.objects[object_id]
                new_x, new_y = input_centroids[col]

                # Exponential Moving Average velocity estimation
                vx = 0.4 * self.velocities[object_id][0] + 0.6 * (new_x - prev_x)
                vy = 0.4 * self.velocities[object_id][1] + 0.6 * (new_y - prev_y)

                self.velocities[object_id] = (vx, vy)
                self.objects[object_id] = (new_x, new_y)
                self.boxes[object_id] = detections[col][:4]
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
                if len(self.objects) < 2:
                    self.register(input_centroids[col], detections[col])

        return self.get_results()

    def get_results(self):
        results = {}
        for obj_id, (cx, cy) in self.objects.items():
            vx, vy = self.velocities.get(obj_id, (0.0, 0.0))
            heading = (math.degrees(math.atan2(vx, -vy)) + 360) % 360
            speed_norm = 10.0
            rel_speed_u = min(30.0, math.sqrt(vx**2 + vy**2) * speed_norm)

            results[obj_id] = {
                "cx": cx,
                "cy": cy,
                "bbox": self.boxes.get(obj_id, (cx - 10, cy - 10, cx + 10, cy + 10)),
                "heading": round(heading, 1),
                "rel_speed_u": round(rel_speed_u, 1)
            }
        return results


def load_altitude_config(config_path):
    if not os.path.exists(config_path):
        return {"camera": {"height_m": 1200.0, "vfov_deg": 40.0}, "landmark": {"pixel_y": 360}}
    with open(config_path, "r") as f:
        return json.load(f)


def estimate_altitude(cy, frame_h, config):
    cam_height = config["camera"]["height_m"]
    vfov = config["camera"]["vfov_deg"]
    
    norm_y = (cy - (frame_h / 2.0)) / (frame_h / 2.0)
    angle_rad = math.radians(-norm_y * (vfov / 2.0))
    
    estimated_alt = cam_height + (angle_rad * 450.0)
    return round(max(50.0, min(5000.0, estimated_alt)), 1)