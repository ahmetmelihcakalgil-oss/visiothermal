"""Run the pipeline on a live RTSP stream.

    python live.py --rtsp rtsp://user:pass@host/stream --empty empty_room.jpg

NOT tested against a real camera: the demo uses synthetic frames. Frames are
processed in memory and never written to disk; only counts and BMS writes
are logged.
"""
import argparse
import time

import cv2

import detector as detmod
from pipeline import Pipeline
from bms import MockBMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", required=True)
    ap.add_argument("--empty", required=True, help="photo of the empty hall, for calibration")
    ap.add_argument("--one-person", help="optional photo with exactly one person seated")
    ap.add_argument("--detector", default="blob", choices=["blob", "yolo"])
    ap.add_argument("--interval", type=float, default=60.0, help="seconds per control step")
    args = ap.parse_args()

    det = detmod.YoloDetector() if args.detector == "yolo" else detmod.BlobDetector()
    empty = cv2.imread(args.empty)
    one = cv2.imread(args.one_person) if args.one_person else None
    det.calibrate(empty, one)
    h, w = empty.shape[:2]

    pipe = Pipeline(det, MockBMS())
    cap = cv2.VideoCapture(args.rtsp)
    next_t = 0.0
    while True:
        ok = cap.grab()
        if ok and time.time() < next_t:
            continue                      # drop frames between control steps
        ok, frame = cap.retrieve() if ok else (False, None)
        if not ok:
            frame = None                  # fail-safe: controller falls back to 100%
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(args.rtsp)
        minute = int(time.time() // 60)
        d, boxes, raw = pipe.process(frame, minute, width=w, height=h)
        print("%s count=%s level=%d setpoint=%.1f fan=%d%% lights=%s%s" % (
            time.strftime("%H:%M:%S"), pipe.last_count, d.level, d.setpoint_c, d.fan_pct,
            d.lights_pct, "  [FALLBACK]" if d.fallback else ""))
        del frame                         # nothing keeps the image
        next_t = time.time() + args.interval


if __name__ == "__main__":
    main()
