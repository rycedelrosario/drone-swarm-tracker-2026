import cv2
import numpy as np


class BlobDetector:
    def __init__(self, min_area=16, max_area=80000, iou_threshold=0.4, max_boxes=80):
        self.min_area = min_area
        self.max_area = max_area
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, mask_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, mask_bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, self.kernel)
        mask_bright = cv2.morphologyEx(mask_bright, cv2.MORPH_OPEN, self.kernel)

        boxes = self._boxes_from_mask(mask_dark) + self._boxes_from_mask(mask_bright)
        boxes = self._nms(boxes)
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes[: self.max_boxes]

    def _boxes_from_mask(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            confidence = min(1.0, area / (w * h)) if w * h > 0 else 0.0
            boxes.append((x, y, x + w, y + h, confidence))
        return boxes

    def _nms(self, boxes):
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < self.iou_threshold]
        return keep

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2, _ = a
        bx1, by1, bx2, by2, _ = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
