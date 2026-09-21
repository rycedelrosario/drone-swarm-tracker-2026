# BlobDetector vs YoloDetector

Comparison run: `scripts/compare_detectors.py`, frames 3000-3120 (~4s) of `data/perdix_swarm_demo.mp4`, same `MultiObjectTracker` settings for both.

| | BlobDetector | YoloDetector (yolov8n, conf=0.12) |
|---|---|---|
| Tracks created | 39 | 8 |
| Avg track length (frames) | 105.1 | 59.1 |
| Approx FPS | 482.6 | 43.7 |

## Which one actually finds the drones

Neither detector knows what a "drone" is in any real sense — blob thresholds on contrast, and YOLOv8n (trained on COCO, which has no drone class) labels these objects `airplane` or `kite`. But eyeballing debug frames from both:

- **Tactical-screen / open-sky frame (frame 5500, two faint drones against blue sky):** BlobDetector found nothing where the drones actually are — it boxed ~9 spots of compression noise in the empty sky and missed both real targets, because they're lower-contrast than the noise Otsu was picking up. YOLO found exactly 2 boxes, tightly centered on both real drones (`kite`, conf ~0.33).
- **Jets-against-sky frame (frame 500):** YOLO correctly boxed the clearly-visible jet (`airplane`, conf 0.76) but missed the second jet that's mostly cut off at the top edge of frame, throwing a false positive on a contrail streak instead (conf 0.47). BlobDetector isn't directly comparable here since it doesn't reason about object identity at all — it just boxes every sufficiently dark/bright blob, jets included.
- **Title-card frame (first_frame.jpg, pure UI text/logo, no aircraft):** BlobDetector boxed all 64 high-contrast blobs it found — the seal graphic and literally every letter of the on-screen text. YOLO (not shown here) would find zero, since none of that resembles its trained classes.

## What each gets wrong

**BlobDetector:**
- No notion of object identity — anything with enough local contrast becomes a box: text, logos, compression artifacts, sensor noise.
- Fails exactly when the real target is *lower* contrast than the surrounding noise (frame 5500), which is common for small distant drones.
- This inflates track count (39 vs 8 in the same window) with short-lived, spurious tracks that the tracker has to churn through.

**YoloDetector:**
- Confidence on the actual drones is barely above the 0.12 threshold (~0.33) — they're tiny and don't look like anything in COCO's training set, so raising the threshold at all would likely drop real detections.
- Misses partially-out-of-frame or heavily occluded aircraft (the second jet in frame 500) and can still hallucinate on ambiguous shapes like contrails.
- Far fewer total tracks (8 vs 39), which is a mix of genuinely fewer false positives *and* real misses on some frames — some of blob's "extra" tracks in this window may be real objects blob caught that YOLO's confidence threshold filtered out.

## FPS

BlobDetector is ~11x faster (482 vs 44 fps) since it's just thresholding + contour extraction — no neural network. YoloDetector still ran well above the "under 1fps on CPU" expectation from the course notes, which suggests this machine picked up GPU/MPS acceleration automatically; on a CPU-only machine the gap would be far larger.

## Takeaway

BlobDetector is a noisy but exhaustive net — it catches things YOLO won't (and can't, since YOLO has no "drone" class), at the cost of many false positives. YoloDetector is far more precise on real aircraft-shaped objects but is blind to anything that doesn't resemble its training classes, and its confidence on the actual drones is uncomfortably close to the detection threshold.
