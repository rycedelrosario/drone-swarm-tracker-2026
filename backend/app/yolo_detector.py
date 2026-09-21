from pathlib import Path

from ultralytics import YOLO

LOCAL_WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "yolov8n.pt"


class YoloDetector:
    def __init__(self, conf_threshold=0.12):
        weights = str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else "yolov8n.pt"
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def detect(self, frame_bgr):
        results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))
        return boxes
