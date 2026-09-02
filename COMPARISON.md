# Detector Comparison: Otsu Blob Detector vs. YOLOv8n

## Overview
This document compares the performance and trade-offs of the traditional Computer Vision Blob Detector (Otsu thresholding + contour analysis) against the Deep Learning YOLOv8n object detector on the `perdix_demo.mp4` aerial swarm video dataset.

## Comparison Summary

| Metric / Feature | Otsu Blob Detector | YOLOv8n Model |
| :--- | :--- | :--- |
| **Detection Speed (FPS)** | Fast (~30–60 FPS on CPU) | Moderate (~5–15 FPS on CPU) |
| **False Positives** | Higher (detects clouds/reflections) | Lower (filters non-object noise better) |
| **Small Object Sensitivity** | Very High (picks up sub-pixel specks) | Requires low confidence threshold (`0.12`) |
| **Resource Usage** | Lightweight (minimal CPU & memory) | Requires PyTorch & model weights (~6MB) |

## Observations
1. **Otsu Blob Detector:**
   - Detects high volumes of tiny airborne targets efficiently without GPU acceleration.
   - Susceptible to background clutter and brightness variations in aerial video cuts.

2. **YOLOv8n Detector:**
   - Provides cleaner target bounding boxes around larger or medium-range targets.
   - Small drone specks against the sky require setting a reduced confidence threshold (`conf=0.12`) to avoid missing distant targets.