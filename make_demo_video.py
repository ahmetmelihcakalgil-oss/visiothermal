"""Render a recording of the Python demo run.

The video is produced by actually running the synthetic day through the same
Pipeline (detector -> median filter -> controller -> MockBMS) and the energy
model, one simulated minute per frame. Nothing is screen-captured; every
frame shows the pipeline's live output, so the clip always matches the code.

    python make_demo_video.py [--out out/demo_run.mp4] [--fps 24]
"""
import argparse
import os
import subprocess

import cv2
import numpy as np

import synth
import detector as detmod
import energy
from pipeline import Pipeline
from bms import MockBMS
from controller import MAX_CFM, LIGHT_ACTIVE

W, H = 1280, 720
FPS = 24
ROOM = (720, 480)
ROOM_X, ROOM_Y = 16, 60
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 752, 60, 512, 480
TL_X, TL_Y, TL_W, TL_H = 16, 556, 1248, 148

BG = (32, 18, 11)          # BGR #0b1220
PANEL = (46, 28, 17)
LINE = (74, 48, 34)
TEXT = (247, 237, 230)
MUTED = (189, 164, 147)
TEAL = (166, 184, 20)
ORANGE = (11, 158, 245)
RED = (60, 60, 220)
GRAY = (164, 168, 154)

FAIL_START, FAIL_END = 14 * 60 + 5, 14 * 60 + 12


def hhmm(m):
    return "%02d:%02d" % (int(m) // 60, int(m) % 60)


def _txt(img, s, xy, size=0.5, color=TEXT, thick=1, align="l", font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(s, font, size, thick)
    x, y = xy
    if align == "r":
        x -= tw
    elif align == "c":
        x -= tw // 2
    cv2.putText(img, s, (x, y), font, size, color, thick, cv2.LINE_AA)
    return tw


def _row(img, y, label, value, color=TEXT):
    _txt(img, label, (16, y), 0.46, MUTED)
    _txt(img, value, (PANEL_W - 16, y), 0.55, color, 1, "r")


def _divider(img, y):
    cv2.line(img, (14, y), (PANEL_W - 14, y), LINE, 1)


def room_image(frame, boxes, decision, minute, count):
    out = cv2.resize(frame, ROOM, interpolation=cv2.INTER_LINEAR)
    sx, sy = ROOM[0] / float(synth.W), ROOM[1] / float(synth.H)

    heat = np.zeros((synth.H, synth.W), np.float32)
    for x, y, w, h in boxes:
        cv2.circle(heat, (int(x + w / 2), int(y + h / 2)), 26, 1.0, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), 14)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat = cv2.resize(heat, ROOM)
    cm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (heat * 0.45)[..., None]
    out = (out * (1 - alpha) + cm * alpha).astype(np.uint8)

    y0, y1 = 40, synth.H - 20
    names = ("on", "orta", "arka")
    for b in range(3):
        by0 = int((y0 + b * (y1 - y0) / 3.0) * sy)
        by1 = int((y0 + (b + 1) * (y1 - y0) / 3.0) * sy)
        lp = decision.lights_pct[b]
        if lp < LIGHT_ACTIVE:
            out[by0:by1] = (out[by0:by1].astype(np.float32) * 0.55).astype(np.uint8)
        _txt(out, "isik %s %%%d" % (names[b], lp),
             (ROOM[0] - 10, by0 + 20), 0.46, (255, 255, 255), 1, "r")

    for x, y, w, h in boxes:
        cv2.rectangle(out, (int(x * sx), int(y * sy)),
                      (int((x + w) * sx), int((y + h) * sy)), (0, 220, 130), 2)

    cv2.rectangle(out, (0, 0), (ROOM[0], 30), (28, 24, 20), -1)
    _txt(out, "%s   sayilan %d / %d   seviye %%%d" %
         (hhmm(minute), count, synth.CAPACITY, decision.level),
         (10, 21), 0.62, TEXT, 1)
    if decision.fallback:
        cv2.rectangle(out, (0, 0), (ROOM[0], ROOM[1]), (0, 0, 120), -1)
        cv2.addWeighted(out, 0.45, np.zeros_like(out), 0.55, 0, out)
        _txt(out, "KAMERA YOK", (ROOM[0] // 2, ROOM[1] // 2 - 10), 1.2, TEXT, 2, "c")
        _txt(out, "guvenli mod: eski program (%100)", (ROOM[0] // 2, ROOM[1] // 2 + 24),
             0.6, TEXT, 1, "c")
    return out


def panel(decision, count, true_n, minute, hv_b, hv_o, li_b, li_o, last_bms, outage):
    p = np.full((PANEL_H, PANEL_W, 3), PANEL, np.uint8)
    y = 28
    _txt(p, "CANLI DURUM", (16, y), 0.5, TEAL, 1)
    y += 28
    _row(p, y, "Saat", hhmm(minute)); y += 24
    _row(p, y, "Sayilan / gercek", "%s / %d" % ("-" if outage else count, true_n)); y += 24
    _row(p, y, "Doluluk", "%%%d" % (0 if outage else round(100.0 * count / synth.CAPACITY))); y += 24
    _row(p, y, "HVAC seviyesi", "%%%d" % decision.level, TEAL if not outage else RED); y += 24
    _row(p, y, "Setpoint", "%.1f C" % decision.setpoint_c); y += 24
    _row(p, y, "Fan", "%%%d - %d CFM" % (decision.fan_pct, decision.fan_cfm)); y += 24
    _row(p, y, "Isik on/orta/arka", " / ".join("%d" % v for v in decision.lights_pct)); y += 18
    _divider(p, y); y += 24
    hvac_kw = energy.optimized_kw(energy.Plant(), decision.level, decision.lights_pct)[0]
    _row(p, y, "HVAC gucu", "%.1f kW" % hvac_kw, TEAL); y += 24
    _row(p, y, "Baz durum", "%.1f kW" % energy.Plant().hvac_kw, ORANGE); y += 24
    _row(p, y, "Anlik tasarruf", "%.1f kW" % (energy.Plant().hvac_kw - hvac_kw),
         TEAL if hvac_kw < energy.Plant().hvac_kw else MUTED); y += 18
    _divider(p, y); y += 24
    total_b = hv_b + li_b
    total_o = hv_o + li_o
    _row(p, y, "Enerji bugun (opt / baz)", "%.1f / %.1f kWh" % (total_o, total_b)); y += 24
    saved = 100.0 * (total_b - total_o) / total_b if total_b else 0.0
    _row(p, y, "Tasarruf", "%%%.1f" % saved, TEAL); y += 18
    _divider(p, y); y += 24
    _row(p, y, "BMS son yazim", last_bms or "-"); y += 24
    _row(p, y, "Kamera", "KESINTI" if outage else "normal", RED if outage else TEXT)

    log = "[%s] count=%s level=%d set=%.1f fan=%d%%" % (
        hhmm(minute), "-" if outage else count, decision.level, decision.setpoint_c,
        decision.fan_pct)
    cv2.rectangle(p, (10, PANEL_H - 34), (PANEL_W - 10, PANEL_H - 8), BG, -1)
    _txt(p, log, (16, PANEL_H - 14), 0.44, TEAL, 1, "l", cv2.FONT_HERSHEY_DUPLEX)
    return p


def timeline(t, true, det, hv_base, hv_opt, level, fallback, cur):
    p = np.full((TL_H, TL_W, 3), PANEL, np.uint8)
    n = len(t)
    L, R = 42, 10
    x_at = lambda i: L + (TL_W - L - R) * i / max(1, n - 1)

    # occupancy
    y0, y1 = 8, 58
    yv = lambda v: y1 - (y1 - y0) * v / float(synth.CAPACITY)
    cv2.line(p, (L, y1), (TL_W - R, y1), LINE, 1)
    _txt(p, "kisi", (TL_W - R, y0 + 10), 0.4, MUTED, 1, "r")
    for v in (0, synth.CAPACITY):
        _txt(p, str(v), (L - 6, int(yv(v)) + 4), 0.36, MUTED, 1, "r")
    pts = np.array([[x_at(i), yv(v)] for i, v in enumerate(true[:cur + 1])], np.int32)
    if len(pts) > 1:
        cv2.polylines(p, [pts], False, GRAY, 1, cv2.LINE_AA)
    pts = np.array([[x_at(i), yv(v)] for i, v in enumerate(det[:cur + 1]) if v is not None],
                   np.int32)
    if len(pts) > 1:
        cv2.polylines(p, [pts], False, TEAL, 2, cv2.LINE_AA)

    # power
    p0, p1 = 70, 118
    yw = lambda v: p1 - (p1 - p0) * v / 16.0
    cv2.line(p, (L, p1), (TL_W - R, p1), LINE, 1)
    _txt(p, "kW", (TL_W - R, p0 + 10), 0.4, MUTED, 1, "r")
    for v in (0, 16):
        _txt(p, str(v), (L - 6, int(yw(v)) + 4), 0.36, MUTED, 1, "r")
    base_pts = np.array([[x_at(i), yw(v)] for i, v in enumerate(hv_base[:cur + 1])], np.int32)
    opt_pts = np.array([[x_at(i), yw(v)] for i, v in enumerate(hv_opt[:cur + 1])], np.int32)
    if len(opt_pts) > 1:
        poly = np.vstack([opt_pts, [[opt_pts[-1][0], p1], [opt_pts[0][0], p1]]])
        cv2.fillPoly(p, [poly], (60, 46, 20))
        cv2.polylines(p, [opt_pts], False, TEAL, 1, cv2.LINE_AA)
    if len(base_pts) > 1:
        cv2.polylines(p, [base_pts], False, ORANGE, 1, cv2.LINE_AA)

    # level strip
    for i in range(cur + 1):
        col = RED if fallback[i] else {0: (160, 163, 148), 25: (183, 231, 110),
                                      50: (166, 184, 20), 100: (110, 118, 15)}[level[i]]
        x = x_at(i)
        cv2.line(p, (int(x), 126), (int(x), 140), col, 2)
    _txt(p, "seviye", (8, 138), 0.4, MUTED)

    # x axis + cursor
    for m in range(synth.OPEN_MIN, synth.CLOSE_MIN + 1, 120):
        i = min(n - 1, m - synth.OPEN_MIN)
        _txt(p, hhmm(m), (int(x_at(i)), TL_H - 2), 0.36, MUTED, 1, "c")
    cx = int(x_at(cur))
    cv2.line(p, (cx, 4), (cx, TL_H - 12), (230, 230, 230), 1)
    return p


def draw_frame(minute, frame, boxes, decision, count, true_n, hv_b, hv_o, li_b, li_o,
               last_bms, outage, rec, cur):
    img = np.full((H, W, 3), BG, np.uint8)
    img[ROOM_Y:ROOM_Y + ROOM[1], ROOM_X:ROOM_X + ROOM[0]] = room_image(
        frame, boxes, decision, minute, 0 if outage else count)
    img[PANEL_Y:PANEL_Y + PANEL_H, PANEL_X:PANEL_X + PANEL_W] = panel(
        decision, count, true_n, minute, hv_b, hv_o, li_b, li_o, last_bms, outage)
    img[TL_Y:TL_Y + TL_H, TL_X:TL_X + TL_W] = timeline(
        rec["t"], rec["true"], rec["det"], rec["hvac_base"], rec["hvac_opt"],
        rec["level"], rec["fallback"], cur)

    cv2.rectangle(img, (0, 0), (W, 40), (24, 14, 8), -1)
    _txt(img, "VisioThermal AI  -  python run_demo.py", (16, 26), 0.6, TEXT, 1)
    cv2.circle(img, (W - 250, 20), 5, (60, 60, 220), -1)
    _txt(img, "REC", (W - 240, 26), 0.5, (60, 60, 220), 1)
    secs = cur / float(FPS)
    _txt(img, "%02d:%02d / 00:30" % (int(secs) // 60, int(secs) % 60),
         (W - 16, 26), 0.5, MUTED, 1, "r")
    _txt(img, "SIMULASYON - sentetik amfi - 1 gun (08:00-20:00)", (W - 300, 26),
         0.46, MUTED, 1, "r")
    return img


def summary_card(img, accuracy, hv_b, hv_o, li_b, li_o, events, fallback_min):
    overlay = img.copy()
    cv2.rectangle(overlay, (ROOM_X, ROOM_Y), (W - 16, ROOM_Y + ROOM[1]), (18, 10, 6), -1)
    img = cv2.addWeighted(overlay, 0.95, img, 0.05, 0)
    _txt(img, "DEMO TAMAMLANDI", (ROOM_X + 28, ROOM_Y + 56), 1.0, TEAL, 2)
    lines = [
        ("islenen kare", str(accuracy["frames"])),
        ("sayim hatasi (kare basina)", "%.2f kisi" % accuracy["mae_people"]),
        ("+/-2 kisi icinde", "%%%.0f" % accuracy["within_2_pct"]),
        ("HVAC", "%.1f -> %.1f kWh  (%%%.1f tasarruf)" %
         (hv_b, hv_o, 100.0 * (hv_b - hv_o) / hv_b)),
        ("Aydinlatma", "%.1f -> %.1f kWh  (%%%.1f tasarruf)" %
         (li_b, li_o, 100.0 * (li_b - li_o) / li_b)),
        ("BMS degisim olayi", str(events)),
        ("kamera kesintisi", "%s-%s - %d dk guvenli mod" %
         (hhmm(FAIL_START), hhmm(FAIL_END), fallback_min)),
    ]
    y = ROOM_Y + 110
    for label, value in lines:
        _txt(img, label, (ROOM_X + 28, y), 0.55, MUTED)
        _txt(img, value, (W - 40, y), 0.55, TEXT, 1, "r")
        cv2.line(img, (ROOM_X + 28, y + 10), (W - 40, y + 10), (50, 32, 22), 1)
        y += 42
    return img


def _ffmpeg_writer(path, fps, size, crf=26):
    """H.264 via the ffmpeg binary bundled with imageio-ffmpeg (if present)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (size[0], size[1]), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class Writer:
    """Frame sink shared by the video builders: H.264 when ffmpeg is available,
    VP8/WebM through OpenCV otherwise (both play in Chromium browsers)."""

    def __init__(self, path, fps, size, crf=26):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.size = path, size
        self.proc = _ffmpeg_writer(path, fps, size, crf)
        self._writer = None
        if self.proc is None:
            self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"),
                                           fps, size)
            if not self._writer.isOpened():
                raise RuntimeError("cannot open video writer for %s" % path)

    def write(self, img):
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(img).astype(np.uint8).tobytes())
        else:
            self._writer.write(img)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding %s" % self.path)
        else:
            self._writer.release()
        return self.path


def build(path, fps=FPS, verbose=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rng = np.random.default_rng(7)
    sched = synth.DaySchedule(seed=7)
    plant = energy.Plant()

    det = detmod.BlobDetector()
    det.calibrate(synth.empty_frame(),
                  synth.render([5 * synth.COLS + 4], np.random.default_rng(0), noise=0))
    bms = MockBMS()
    pipe = Pipeline(det, bms)

    rec = {k: [] for k in ("t", "true", "det", "level", "hvac_base", "hvac_opt",
                           "fallback")}
    hv_b = hv_o = li_b = li_o = 0.0
    last_bms = ""

    out = Writer(path, fps, (W, H))
    write = out.write

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
        rec["det"].append(pipe.last_count)
        rec["level"].append(decision.level)
        rec["hvac_base"].append(b_hvac)
        rec["hvac_opt"].append(round(hvac_kw, 3))
        rec["fallback"].append(bool(decision.fallback))
        if bms.events:
            e = bms.events[-1]
            last_bms = "%s = %s" % (e["object"], e["value"])

        cur = len(rec["t"]) - 1
        write(draw_frame(minute, frame, boxes, decision, pipe.last_count,
                                int(len(seats)), hv_b, hv_o, li_b, li_o, last_bms,
                                outage, rec, cur))
        if verbose and minute % 60 == 0:
            print("  video %s" % hhmm(minute), flush=True)

    # detection accuracy / fallback stats for the closing card
    errs = [abs(d - tr) for d, tr in zip(rec["det"], rec["true"]) if d is not None]
    accuracy = {
        "frames": len(errs),
        "mae_people": float(np.mean(errs)) if errs else 0.0,
        "within_2_pct": 100.0 * np.mean([e <= 2 for e in errs]) if errs else 0.0,
    }
    fallback_min = int(sum(rec["fallback"]))
    last = draw_frame(synth.CLOSE_MIN - 1, frame, [], decision, pipe.last_count, 0,
                      hv_b, hv_o, li_b, li_o, last_bms, False, rec, len(rec["t"]) - 1)
    last = summary_card(last, accuracy, hv_b, hv_o, li_b, li_o, len(bms.events),
                        fallback_min)
    for _ in range(int(fps * 2.5)):
        write(last)
    out.close()
    if verbose:
        print("video written: %s (%.1f MB)" % (path, os.path.getsize(path) / 1048576.0))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("out", "demo_run.mp4"))
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()
    build(a.out, a.fps)
