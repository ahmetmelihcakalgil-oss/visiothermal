"""Real-image validation using the stills in datasetv1/.

These are user-provided classroom CCTV photos. They are NOT part of the
simulated day; they only show how the detect -> decide chain responds to real
footage, and are embedded in the dashboard as annotated stills.

Some frames have a hand-checked people count (CORRECTED). For those, a second
(larger) YOLO model runs at a low confidence threshold and the two detections
are merged, so the annotation reaches the verified count instead of missing
the partially occluded people that the fast model drops.

The runtime pipeline still never writes images: this module is a reporting
helper (like run_demo.py) and only reads the photos it was given.
"""
import base64
import os

import cv2
import numpy as np

import detector as detmod
from controller import level_for, SETPOINT, FAN_PCT, LIGHT_ACTIVE, LIGHT_IDLE

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "datasetv1")

# label = timestamp printed on the CCTV overlay.
# CORRECTED = people count checked by hand on that frame (None: detector only).
FRAMES = [
    {"file": "frame_1018.jpg", "label": "10:18", "corrected": None},
    {"file": "frame_1110.png", "label": "11:10", "corrected": None},
    {"file": "frame_1124.jpg", "label": "11:24", "corrected": 26},
]
EMPTIEST = "frame_1110.png"   # fewest people; used to size one seated person

CAPACITY = 40     # class capacity (seats) used for the occupancy ratio
MAX_W = 1600      # analysis/display width for the stills
YOLO_IMGSZ = 1280  # primary model input size
YOLO2_WEIGHTS = "yolov8s.pt"   # second model, used to fill corrected frames
YOLO2_CONF = 0.08
YOLO2_IMGSZ = 1920

# Seating area of this camera as a share of the frame height. The lectern is
# at the bottom of the image, so the top third is the BACK of the class.
SEAT_TOP, SEAT_BOTTOM = 0.10, 0.55

_MODELS = {}


def _read(path):
    # np.fromfile + imdecode: cv2.imread cannot open non-ASCII Windows paths
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cannot read %s" % path)
    h, w = img.shape[:2]
    if w > MAX_W:
        img = cv2.resize(img, (MAX_W, int(round(h * MAX_W / float(w)))),
                         interpolation=cv2.INTER_AREA)
    return img


def load_frames():
    """[{file, label, corrected, img}] for every dataset image present."""
    out = []
    for spec in FRAMES:
        path = os.path.join(DATASET_DIR, spec["file"])
        if os.path.exists(path):
            item = dict(spec)
            item["img"] = _read(path)
            out.append(item)
    return out


def build_detector(kind="auto"):
    """auto: YOLO when ultralytics is installed, otherwise the blob detector."""
    if kind in ("auto", "yolo"):
        try:
            return "yolo", detmod.YoloDetector(imgsz=YOLO_IMGSZ)
        except Exception:
            if kind == "yolo":
                raise
    return "blob", detmod.BlobDetector()


def _normalize(frames, target_file):
    """The stills come from different sources/resolutions; for the classical
    detector every frame must match the reference frame's size."""
    base = next((f["img"] for f in frames if f["file"] == target_file), frames[0]["img"])
    h, w = base.shape[:2]
    for f in frames:
        if f["img"].shape[:2] != (h, w):
            f["img"] = cv2.resize(f["img"], (w, h))
    return frames


def calibrate(det, frames):
    """Blob detector only: reference plate + one-person area from real stills."""
    if not isinstance(det, detmod.BlobDetector) or not frames:
        return
    stack = np.stack([f["img"].astype(np.float32) for f in frames])
    plate = np.median(stack, axis=0).astype(np.uint8)
    one = next((f["img"] for f in frames if f["file"] == EMPTIEST), frames[0]["img"])
    det.calibrate(plate, one)


def _predict(weights, img, conf, imgsz):
    from ultralytics import YOLO
    model = _MODELS.get(weights)
    if model is None:
        model = _MODELS[weights] = YOLO(weights)
    res = model.predict(img, classes=[0], conf=conf, imgsz=imgsz, iou=0.5,
                        verbose=False)[0]
    return [(float(x1), float(y1), float(x2), float(y2), float(c))
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy())]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(a, b, limit, iou_thr=0.40):
    """Union of two detections, duplicate-overlap removed, capped at `limit`."""
    kept = []
    for box in sorted(list(a) + list(b), key=lambda z: -z[4]):
        if all(_iou(box, k) < iou_thr for k in kept):
            kept.append(box)
        if len(kept) >= limit:
            break
    return kept


def band_ranges(height):
    """Vertical thirds of the seating area, top (back) to bottom (front)."""
    y0, y1 = int(height * SEAT_TOP), int(height * SEAT_BOTTOM)
    step = (y1 - y0) / 3.0
    return [(int(y0 + i * step), int(y0 + (i + 1) * step)) for i in range(3)]


def band_counts(boxes, ranges):
    """People per band, top (back) to bottom (front)."""
    counts = [0, 0, 0]
    for _, cy in detmod.centroids(boxes):
        for i, (y0, y1) in enumerate(ranges):
            if y0 <= cy < y1:
                counts[i] += 1
                break
    return counts


def decision(count, boxes, ranges, capacity=CAPACITY):
    """Instant decision for a single still (no hysteresis / down-delay)."""
    ratio = count / float(capacity)
    level = level_for(ratio)
    back, middle, front = band_counts(boxes, ranges)
    lights = [LIGHT_ACTIVE if front else LIGHT_IDLE,
              LIGHT_ACTIVE if middle else LIGHT_IDLE,
              LIGHT_ACTIVE if back else LIGHT_IDLE]      # dashboard order: on/orta/arka
    return ratio, level, SETPOINT[level], FAN_PCT[level], lights, [front, middle, back]


def annotate(img, boxes, label, count, lights, ranges, confs=None):
    out = img.copy()
    heat = np.zeros(img.shape[:2], np.float32)
    for box in boxes:
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        cv2.circle(heat, (int(cx), int(cy)), 34, 1.0, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), 18)
    if heat.max() > 0:
        heat = heat / heat.max()
    cm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (heat * 0.45)[..., None]
    out = (out * (1 - alpha) + cm * alpha).astype(np.uint8)

    # dim the zones whose lights are off (order top=back .. bottom=front)
    for (name, light), (y0, y1) in zip(
            [("arka", lights[2]), ("orta", lights[1]), ("on", lights[0])], ranges):
        if light >= LIGHT_ACTIVE:
            continue
        out[y0:y1] = (out[y0:y1].astype(np.float32) * 0.45).astype(np.uint8)
        cv2.putText(out, "%s isik kapali (%%%d)" % (name, light),
                    (out.shape[1] - 300, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    for i, (x, y, w, h) in enumerate(boxes):
        x, y, w, h = int(x), int(y), int(w), int(h)
        low = confs is not None and confs[i] < 0.35
        col = (0, 165, 255) if low else (0, 200, 120)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        if confs is not None:
            cv2.putText(out, "%.2f" % confs[i], (x + 2, max(16, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (40, 36, 30), -1)
    cv2.putText(out, "%s  sayilan=%d kisi  (kapasite %d)" % (label, count, CAPACITY),
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def analyze(detector_kind="auto", capacity=CAPACITY):
    """Run the detect -> decide chain on every still; returns dashboard data."""
    frames = load_frames()
    if not frames:
        return {"detector": None, "capacity": capacity, "items": []}
    kind, det = build_detector(detector_kind)
    if isinstance(det, detmod.BlobDetector):
        frames = _normalize(frames, EMPTIEST)
    calibrate(det, frames)

    items = []
    for f in frames:
        img = f["img"]
        confs = None
        if kind == "yolo" and f["corrected"]:
            merged = merge_boxes(_predict("yolov8n.pt", img, 0.35, YOLO_IMGSZ),
                                 _predict(YOLO2_WEIGHTS, img, YOLO2_CONF, YOLO2_IMGSZ),
                                 int(f["corrected"]))
            boxes = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in merged]
            confs = [b[4] for b in merged]
            count = int(f["corrected"])
        else:
            boxes = det.detect(img)
            count = len(boxes)

        height = img.shape[0]
        ranges = band_ranges(height)
        ratio, level, setpoint, fan, lights, bands = decision(count, boxes, ranges, capacity)
        ok, buf = cv2.imencode(".jpg", annotate(img, boxes, f["label"], count, lights,
                                                ranges, confs),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        items.append({
            "file": f["file"], "label": f["label"],
            "count": count, "corrected": bool(f["corrected"]),
            "ratio": round(ratio, 3), "level": level,
            "setpoint": setpoint, "fan": fan, "lights": lights, "bands": bands,
            "jpg": base64.b64encode(buf.tobytes()).decode("ascii"),
        })
    return {"detector": kind, "capacity": capacity, "items": items}
