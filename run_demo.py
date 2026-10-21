"""Simulate one full lecture-hall day and build the dashboard.

    python run_demo.py [--out out] [--seed 7] [--detector blob|yolo]

Everything is synthetic. The pipeline sees frames only in memory; the only
images written are a few annotated demo snapshots (synthetic people).
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

import synth
import detector as detmod
import energy
import realset
from pipeline import Pipeline
from bms import MockBMS
from controller import LEVELS

FAIL_START, FAIL_END = 14 * 60 + 5, 14 * 60 + 12   # simulated camera outage
SNAP_MINUTES = [9 * 60 + 30, 11 * 60 + 30, 12 * 60 + 50, 14 * 60 + 30, 16 * 60 + 30, 19 * 60]


def hhmm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def annotate(frame, boxes, minute, count, level):
    img = frame.copy()
    heat = np.zeros(frame.shape[:2], np.float32)
    for cx, cy in detmod.centroids(boxes):
        cv2.circle(heat, (int(cx), int(cy)), 26, 1.0, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), 14)
    if heat.max() > 0:
        heat = heat / heat.max()
    cm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (heat * 0.45)[..., None]
    img = (img * (1 - alpha) + cm * alpha).astype(np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 120), 1)
    cv2.rectangle(img, (0, 0), (synth.W, 20), (40, 36, 30), -1)
    cv2.putText(img, "%s  people=%d  HVAC level=%d%%" % (hhmm(minute), count, level),
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_detector(kind):
    if kind == "yolo":
        return detmod.YoloDetector()
    return detmod.BlobDetector()


def load_video(path, name=None):
    """Read a clip and return the base64 payload for the self-contained dashboard."""
    if not path or not os.path.exists(path):
        return None
    name = name or os.path.basename(path)
    with open(path, "rb") as f:
        raw = f.read()
    ext = os.path.splitext(name)[1].lower()
    return {"file": name, "size_mb": round(len(raw) / 1048576.0, 1),
            # VP8/WebM fallback when imageio-ffmpeg is not installed
            "mime": "video/webm" if ext == ".webm" else "video/mp4",
            "data": base64.b64encode(raw).decode("ascii")}


def demo_video(out_dir, mode):
    """Recording of the demo run (out/demo_run.mp4) unless --video none."""
    if mode == "none":
        return None
    path = os.path.join(out_dir, "demo_run.mp4")
    if mode == "rebuild" or not os.path.exists(path):
        import make_demo_video
        make_demo_video.build(path)
    return path


def set_video(out_dir, mode):
    """Annotated detector test over the real set clip (out/set_test.mp4)."""
    if mode == "none":
        return None
    path = os.path.join(out_dir, "set_test.mp4")
    if mode == "rebuild" or not os.path.exists(path):
        import make_set_video
        make_set_video.build(path, verbose=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--detector", default="blob", choices=["blob", "yolo"])
    ap.add_argument("--real-detector", default="auto", choices=["auto", "blob", "yolo"],
                    help="detector for the real stills in datasetv1/ (auto: yolo if installed)")
    ap.add_argument("--video", default="auto", choices=["auto", "rebuild", "none"],
                    help="videos embedded in the dashboard: the demo run recording "
                         "(out/demo_run.mp4) and the real set clip test "
                         "(out/set_test.mp4); auto reuses them when present")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sched = synth.DaySchedule(seed=args.seed)
    plant = energy.Plant()

    det = build_detector(args.detector)
    # one-time site calibration: empty room + one person sitting in the room
    det.calibrate(synth.empty_frame(),
                  synth.render([5 * synth.COLS + 4], np.random.default_rng(0), noise=0))
    bms = MockBMS(path=os.path.join(args.out, "bms_log.jsonl"))
    pipe = Pipeline(det, bms)

    rec = {k: [] for k in ["t", "true", "raw", "det", "level", "setpoint", "fan", "light",
                           "hvac_base", "hvac_opt", "light_opt", "fallback", "pts"]}
    snaps = []
    hv_b = hv_o = li_b = li_o = 0.0

    for minute in range(synth.OPEN_MIN, synth.CLOSE_MIN):
        seats = sched.seats_at(minute)
        gain = 1.0 + 0.06 * np.sin(minute / 53.0) + rng.normal(0, 0.01)
        frame = synth.render(seats, rng, gain=gain)
        outage = FAIL_START <= minute < FAIL_END
        decision, boxes, raw = pipe.process(None if outage else frame, minute)

        hvac_kw, light_kw = energy.optimized_kw(plant, decision.level, decision.lights_pct)
        b_hvac, b_light = energy.baseline_kw(plant)
        hv_b += b_hvac / 60.0
        li_b += b_light / 60.0
        hv_o += hvac_kw / 60.0
        li_o += light_kw / 60.0

        rec["t"].append(minute)
        rec["true"].append(int(len(seats)))
        rec["raw"].append(raw)
        rec["det"].append(pipe.last_count)
        rec["level"].append(decision.level)
        rec["setpoint"].append(decision.setpoint_c)
        rec["fan"].append(decision.fan_pct)
        rec["light"].append(decision.lights_pct)
        rec["hvac_base"].append(b_hvac)
        rec["hvac_opt"].append(round(hvac_kw, 3))
        rec["light_opt"].append(round(light_kw, 3))
        rec["fallback"].append(bool(decision.fallback))
        flat = []
        for cx, cy in detmod.centroids(boxes):
            flat += [int(cx), int(cy)]
        rec["pts"].append(flat)

        if minute in SNAP_MINUTES and not outage:
            img = annotate(frame, boxes, minute, pipe.last_count, decision.level)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            snaps.append({"t": minute, "count": pipe.last_count, "true": int(len(seats)),
                          "level": decision.level,
                          "jpg": base64.b64encode(buf.tobytes()).decode("ascii")})

    bms.flush()

    # detection accuracy, excluding the simulated outage
    errs, serrs = [], []
    for tr, rw, dt in zip(rec["true"], rec["raw"], rec["det"]):
        if rw is None:
            continue
        errs.append(abs(rw - tr))      # per-frame detector error
        serrs.append(abs(dt - tr))     # after the 3-frame median filter
    errs, serrs = np.array(errs), np.array(serrs)
    idle_min = sum(1 for tr in rec["true"] if tr / synth.CAPACITY < 0.10)
    accuracy = {
        "frames": int(len(errs)),
        "mae_people": float(errs.mean()),
        "max_err_people": int(errs.max()),
        "within_2_pct": float(100.0 * (errs <= 2).mean()),
        "smoothed_mae_people": float(serrs.mean()),
        "smoothed_max_err_people": int(serrs.max()),
        "capacity": synth.CAPACITY,
        "idle_hours": idle_min / 60.0,
        "fallback_minutes": int(sum(rec["fallback"])),
    }
    summary = energy.summarize(plant, hv_b, hv_o, li_b, li_o)
    real = realset.analyze(args.real_detector)   # real stills in datasetv1/
    video = load_video(demo_video(args.out, args.video), "demo_run.mp4")
    setvid = load_video(set_video(args.out, args.video), "set_test.mp4")

    data = {
        "meta": {"width": synth.W, "height": synth.H, "rows": synth.ROWS, "cols": synth.COLS,
                 "capacity": synth.CAPACITY, "levels": list(LEVELS),
                 "seats": [[float(x), float(y)] for x, y in synth.SEATS],
                 "detector": args.detector, "seed": args.seed,
                 "fail": [FAIL_START, FAIL_END]},
        "rec": rec, "snaps": snaps, "summary": summary, "accuracy": accuracy,
        "real": real,
        "video": video,
        "setvideo": setvid,
        "bms": bms.events[:400], "bms_total": len(bms.events),
        "plant": {"hvac_kw": plant.hvac_kw, "lighting_kw": plant.lighting_kw,
                  "hours": plant.hours_per_day, "days": plant.academic_days,
                  "halls": plant.halls},
    }
    with open(os.path.join(args.out, "day.json"), "w", encoding="utf-8") as f:
        # the clips are only needed by the HTML (already has them inlined);
        # keeping the base64 here would double the artifact size
        json.dump({k: v for k, v in data.items() if k not in ("video", "setvideo")},
                  f, separators=(",", ":"))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "dashboard_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(os.path.join(args.out, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("detector: %s" % args.detector)
    print("per-frame detector: frames %d  MAE %.2f people  max err %d  within +-2: %.1f%%" % (
        accuracy["frames"], accuracy["mae_people"], accuracy["max_err_people"],
        accuracy["within_2_pct"]))
    print("after 3-frame median: MAE %.2f  max err %d (lag during fast arrivals)" % (
        accuracy["smoothed_mae_people"], accuracy["smoothed_max_err_people"]))
    print("idle (<10%% full): %.1f h   fallback minutes: %d" % (
        accuracy["idle_hours"], accuracy["fallback_minutes"]))
    print("HVAC  baseline %.1f kWh -> %.1f kWh  (%.1f%% saved)" % (
        hv_b, hv_o, summary["hvac_saving_pct"]))
    print("Light baseline %.1f kWh -> %.1f kWh  (%.1f%% saved)" % (
        li_b, li_o, summary["light_saving_pct"]))
    print("BMS change events: %d" % len(bms.events))
    for it in real["items"]:
        print("real still %s (%s): %d people -> level %d%%  set %.1fC  fan %d%%  lights %s" % (
            it["file"], it["label"], it["count"], it["level"], it["setpoint"], it["fan"],
            it["lights"]))


if __name__ == "__main__":
    main()
