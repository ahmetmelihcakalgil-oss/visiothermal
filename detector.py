"""People counting layer.

BlobDetector: classical background subtraction + area-based count. It runs
anywhere (numpy + OpenCV only) and is the prototype default.

YoloDetector: optional drop-in using ultralytics (YOLOv8/v11) when installed.
Both return only bounding boxes; no face or biometric data is produced.
"""
import numpy as np
import cv2


class BlobDetector:
    def __init__(self, diff_thresh=28, min_area_ratio=0.25):
        self.diff_thresh = diff_thresh
        self.min_area_ratio = min_area_ratio
        self.ref = None
        self.person_area = None

    @staticmethod
    def _prep(frame):
        # keep color: luminance alone hides clothes that match the seat brightness
        return cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)

    def calibrate(self, empty_frame, one_person_frame=None, default_area=250.0):
        """Site calibration: one empty-room frame, optionally one with a single person."""
        self.ref = self._prep(empty_frame)
        self.person_area = float(default_area)
        if one_person_frame is not None:
            comps = self._components(one_person_frame)
            if comps:
                self.person_area = float(max(c[4] for c in comps))

    def _components(self, frame):
        g = self._prep(frame)
        # global exposure compensation; people cover well under half the frame
        gain = float(np.median(g)) / max(float(np.median(self.ref)), 1.0)
        diff = np.abs(g - self.ref * gain).max(axis=2)
        mask = (diff > self.diff_thresh).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            out.append((int(x), int(y), int(w), int(h), int(a), float(cent[i][0]), float(cent[i][1])))
        return out

    def detect(self, frame):
        """Return a list of person boxes (x, y, w, h) for this frame."""
        if self.ref is None:
            raise RuntimeError("calibrate() must be called first")
        boxes = []
        for x, y, w, h, a, cx, cy in self._components(frame):
            if a < self.min_area_ratio * self.person_area:
                continue
            n = max(1, int(round(a / self.person_area)))
            if n == 1:
                boxes.append((x, y, w, h))
                continue
            # merged blob: split the box along its longer side into n parts
            if w >= h:
                step = w / n
                boxes += [(int(x + k * step), y, int(step), h) for k in range(n)]
            else:
                step = h / n
                boxes += [(x, int(y + k * step), w, int(step)) for k in range(n)]
        return boxes


class YoloDetector:
    """Optional YOLO backend. Needs `pip install ultralytics` and model weights.

    Note: pretrained COCO weights are trained on side views; for ceiling
    cameras fine-tune on overhead data before relying on the counts.
    """

    def __init__(self, weights="yolov8n.pt", conf=0.35, imgsz=640):
        from ultralytics import YOLO  # imported lazily on purpose
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def calibrate(self, *args, **kwargs):
        return None

    def detect(self, frame):
        res = self.model.predict(frame, classes=[0], conf=self.conf,
                                 imgsz=self.imgsz, verbose=False)[0]
        boxes = []
        for b in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes


def centroids(boxes):
    return [(x + w / 2.0, y + h / 2.0) for x, y, w, h in boxes]


def zone_counts(boxes, width, height, grid=(3, 3)):
    """Occupancy per cell of a grid (rows x cols); the heatmap in the PDR."""
    rows, cols = grid
    z = np.zeros((rows, cols), dtype=int)
    for cx, cy in centroids(boxes):
        r = min(rows - 1, max(0, int(cy / height * rows)))
        c = min(cols - 1, max(0, int(cx / width * cols)))
        z[r, c] += 1
    return z


def band_counts(boxes, height, y0, y1, bands=3):
    """People per horizontal band of the seating area (front/middle/back)."""
    out = [0] * bands
    span = (y1 - y0) / bands
    for _, cy in centroids(boxes):
        k = int((cy - y0) // span)
        out[min(bands - 1, max(0, k))] += 1
    return out
