""""Test video" of the real classroom clip: run the detector over datasetv1.

`datasetv1/demo_1020.mp4` (the user's `1020(1).mp4`) is a 15 s, 450-frame
set of three scenes: 10:18, 11:24 and 11:10. This module plays it through
the same detect -> decide chain used for the real stills (realset.py) and
writes an annotated clip, so the dashboard can show the detector working on
real footage instead of only on three still frames.

Every `stride`-th frame is analysed (YOLOv8n; on the 11:24 scene the second,
larger model is merged in and the result is capped at the hand-checked count
of 26). Frames in between hold the last detections.

    python make_set_video.py [--out out/set_test.mp4]
"""
import os

import cv2
import numpy as np

import realset
from make_demo_video import Writer

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(realset.DATASET_DIR, "demo_1020.mp4")
N_CONF, N_IMGSZ = 0.35, realset.YOLO_IMGSZ
STRIDE = 10          # analysed frames every N; others reuse the detections
CUT_DIFF = 12.0      # mean abs difference (0-255) that marks a scene cut
TOP_BAR, BOT_BAR = 44, 74

BG = (32, 18, 11)          # BGR #0b1220
TEXT = (247, 237, 230)
MUTED = (189, 164, 147)
TEAL = (166, 184, 20)
ORANGE = (11, 158, 245)

SPEC = {f["label"]: f for f in realset.FRAMES}


def _txt(img, s, xy, size=0.5, color=TEXT, thick=1, align="l",
         font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(s, font, size, thick)
    x, y = xy
    if align == "r":
        x -= tw
    elif align == "c":
        x -= tw // 2
    cv2.putText(img, s, (x, y), font, size, color, thick, cv2.LINE_AA)
    return tw


def _sig(img):
    g = cv2.resize(img, (64, 36)).astype(np.float32)
    return g.mean(axis=2)


def scenes(source=SOURCE):
    """[(start, end, label)] - scene cuts found by frame differencing and
    matched to the three stills by appearance."""
    stills = realset.load_frames()
    if not stills:
        raise RuntimeError("no stills in %s to label the scenes" % realset.DATASET_DIR)
    still_sig = [_sig(f["img"]) for f in stills]
    still_label = [f["label"] for f in stills]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % source)
    sigs, cuts = [], []
    prev = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        s = _sig(img)
        if prev is not None and float(np.abs(s - prev).mean()) > CUT_DIFF:
            cuts.append(len(sigs))
        prev = s
        sigs.append(s)
    cap.release()
    if not sigs:
        raise RuntimeError("no frames in %s" % source)

    bounds = [0] + cuts + [len(sigs)]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mid = sigs[(a + b) // 2]
        errs = [float(np.abs(mid - s).mean()) for s in still_sig]
        segs.append([a, b, still_label[int(np.argmin(errs))]])
    # keep the labels unique when the cuts could not be separated
    used = [s[2] for s in segs]
    for f in realset.FRAMES:
        if f["label"] in used:
            continue
        for s in segs:
            if used.count(s[2]) > 1:
                s[2] = f["label"]
                used = [x[2] for x in segs]
                break
    return [(int(a), int(b), lab) for a, b, lab in segs]


def detect(img, label):
    """Detections for one analysed frame (xywh + confidences)."""
    n = realset._predict("yolov8n.pt", img, N_CONF, N_IMGSZ)
    corrected = (SPEC.get(label) or {}).get("corrected")
    if corrected:
        s = realset._predict(realset.YOLO2_WEIGHTS, img, realset.YOLO2_CONF,
                             realset.YOLO2_IMGSZ)
        xyxy = realset.merge_boxes(n, s, int(corrected))
    else:
        xyxy = n
    boxes = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in xyxy]
    return boxes, [b[4] for b in xyxy]


def render(img, boxes, confs, label, scene_i, scene_n, count, ratio, level,
           setpoint, fan, lights, bands, idx, total):
    height = img.shape[0]
    ranges = realset.band_ranges(height)
    out = realset.annotate(img, boxes, label, count, lights, ranges, confs)

    cv2.rectangle(out, (0, 0), (out.shape[1], TOP_BAR), (24, 14, 8), -1)
    _txt(out, "SET VIDEOSU  -  datasetv1/1020(1).mp4  -  dedektor test kaydi",
         (16, 29), 0.6, TEXT, 1)
    _txt(out, "SAHNE %d/%d  %s" % (scene_i + 1, scene_n, label),
         (out.shape[1] - 16, 29), 0.6, TEAL, 1, "r")

    y0 = out.shape[0] - BOT_BAR
    cv2.rectangle(out, (0, y0), (out.shape[1], out.shape[0]), (24, 14, 8), -1)
    _txt(out, "sayilan %d / %d    doluluk %%%d    seviye %%%d    set %.1f C    "
              "fan %%%d" % (count, realset.CAPACITY, round(100 * ratio), level,
                            setpoint, fan),
         (16, y0 + 28), 0.62, TEXT, 1)
    _txt(out, "isik on/orta/arka %s    bantlar %s kisi    "
              "tespit yolov8n%s    kare %d/%d"
         % (" / ".join("%%%d" % v for v in lights),
            "/".join("%d" % b for b in bands),
            " + yolov8s (birlestirme, foto 26)" if SPEC.get(label, {}).get("corrected")
            else "",
            idx + 1, total),
         (16, y0 + 56), 0.5, MUTED, 1, "l", cv2.FONT_HERSHEY_DUPLEX)
    cv2.rectangle(out, (0, out.shape[0] - 5), (out.shape[1], out.shape[0]),
                  (46, 30, 18), -1)
    x = int(out.shape[1] * (idx + 1) / float(total))
    cv2.rectangle(out, (0, out.shape[0] - 5), (x, out.shape[0]), TEAL, -1)
    return out


def build(path, source=SOURCE, stride=STRIDE, crf=24, verbose=True):
    """Detect -> decide over every frame of the set clip; write `path`."""
    segs = scenes(source)
    if verbose:
        print("scenes: " + ", ".join("%s [%d..%d]" % (lab, a, b) for a, b, lab in segs))

    cap = cv2.VideoCapture(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = Writer(path, fps, (w, h), crf=crf)

    boxes, confs, last_seg, counts = [], [], -1, []
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        si = next((i for i, (a, b, _) in enumerate(segs) if a <= idx < b),
                  len(segs) - 1)
        start, _end, label = segs[si]
        if si != last_seg or idx % stride == 0:
            boxes, confs = detect(img, label)
            last_seg = si
        count = len(boxes)
        counts.append(count)
        ranges = realset.band_ranges(h)
        ratio, level, setpoint, fan, lights, bands = realset.decision(
            count, boxes, ranges)
        frame = render(img, boxes, confs, label, si, len(segs), count, ratio,
                       level, setpoint, fan, lights, bands, idx, total)
        out.write(frame)
        idx += 1
        if verbose and idx % 150 == 0:
            print("  scene %d/%d  %s  frames %d  count %d"
                  % (si + 1, len(segs), label, idx, count), flush=True)
    cap.release()
    out.close()
    if verbose:
        for a, b, lab in segs:
            seg_counts = counts[a:b]
            print("  %s: count min %d  max %d  (video %s)"
                  % (lab, min(seg_counts), max(seg_counts), path))
        print("set video written: %s (%.1f MB)"
              % (path, os.path.getsize(path) / 1048576.0))
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("out", "set_test.mp4"))
    ap.add_argument("--stride", type=int, default=STRIDE)
    a = ap.parse_args()
    build(a.out, stride=a.stride)
