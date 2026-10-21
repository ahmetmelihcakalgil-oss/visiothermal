"""Synthetic lecture hall generator with known ground truth.

Renders a ceiling-camera view of a lecture hall (seat grid, people as
shoulder ellipse + head), plus a one-day occupancy schedule. No real
people or real camera data are involved.
"""
import numpy as np
import cv2

W, H = 480, 320
ROWS, COLS = 8, 12
CAPACITY = ROWS * COLS

OPEN_MIN = 8 * 60     # 08:00
CLOSE_MIN = 20 * 60   # 20:00

# (start, end, target fill ratio) as minute-of-day
CLASSES = [
    (8 * 60 + 30, 10 * 60 + 20, 0.88),
    (10 * 60 + 30, 12 * 60 + 20, 0.62),
    (13 * 60 + 30, 15 * 60 + 20, 0.38),
    (15 * 60 + 30, 17 * 60 + 20, 0.92),
]
RAMP_IN = 8    # minutes of arrival before a class starts
RAMP_OUT = 5   # minutes of leaving after a class ends


def seat_centers():
    margin, aisle, top, bottom = 30, 30, 48, 22
    px = (W - 2 * margin - aisle) / COLS
    py = (H - top - bottom) / ROWS
    pts = []
    for r in range(ROWS):
        for c in range(COLS):
            x = margin + px * (c + 0.5) + (aisle if c >= COLS // 2 else 0)
            y = top + py * (r + 0.5)
            pts.append((x, y))
    return np.array(pts, dtype=np.float32)


SEATS = seat_centers()
SEAT_ROW = np.repeat(np.arange(ROWS), COLS)

# BGR colors for clothing; the gray one is close to the chair color on purpose
CLOTHES = [
    (50, 40, 35), (25, 25, 25), (50, 50, 190), (235, 235, 235),
    (120, 170, 200), (70, 140, 70), (180, 90, 40), (60, 60, 60),
    (150, 110, 80), (95, 92, 90),
]
HEADS = [(130, 160, 200), (110, 140, 185), (30, 30, 30), (60, 70, 90)]


def background():
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (176, 182, 188)
    grad = np.linspace(0.94, 1.06, H, dtype=np.float32)[:, None, None]
    img = np.clip(img * grad, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (60, 6), (W - 60, 34), (120, 110, 100), -1)
    for x, y in SEATS:
        cv2.rectangle(img, (int(x - 12), int(y - 4)), (int(x + 12), int(y + 10)),
                      (92, 88, 84), -1)
        cv2.rectangle(img, (int(x - 14), int(y - 12)), (int(x + 14), int(y - 8)),
                      (140, 150, 160), -1)
    return img


_BG = background()


def render(seat_idx, rng, gain=1.0, jitter=2.5, noise=2.5):
    """Render one frame with people seated at the given seat indices."""
    img = _BG.copy()
    for i in seat_idx:
        x, y = SEATS[i] + rng.normal(0, jitter, 2)
        cloth = CLOTHES[rng.integers(len(CLOTHES))]
        head = HEADS[rng.integers(len(HEADS))]
        ang = float(rng.normal(0, 8))
        cv2.ellipse(img, (int(x), int(y + 2)), (11, 6), ang, 0, 360, cloth, -1)
        cv2.circle(img, (int(x), int(y - 1)), 5, head, -1)
    out = img.astype(np.float32) * gain + rng.normal(0, noise, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def empty_frame():
    return _BG.copy()


def _ramp(t, start, end, level):
    if start - RAMP_IN <= t < start:
        return level * (t - (start - RAMP_IN)) / RAMP_IN
    if start <= t <= end:
        return level
    if end < t <= end + RAMP_OUT:
        return level * (1 - (t - end) / RAMP_OUT)
    return 0.0


def target_ratio(t):
    """Planned fill ratio (0..1) at minute-of-day t."""
    r = max(_ramp(t, s, e, lv) for s, e, lv in CLASSES)
    return r


class DaySchedule:
    """Maps minute-of-day to a set of occupied seat indices (stable per class)."""

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        # front-biased seating order, one per class, fixed for the whole class
        self.orders = []
        for _ in CLASSES:
            key = SEAT_ROW + self.rng.normal(0, 1.6, CAPACITY)
            self.orders.append(np.argsort(key))
        self.idle_order = self.rng.permutation(CAPACITY)

    def _class_of(self, t):
        best, best_k = 0.0, None
        for k, (s, e, lv) in enumerate(CLASSES):
            v = _ramp(t, s, e, lv)
            if v > best:
                best, best_k = v, k
        return best_k

    def seats_at(self, t):
        k = self._class_of(t)
        if k is None:
            # idle: cleaners or a student studying alone, 0-3 people
            n = int(2 + 2 * np.sin(t / 37.0)) if t >= 17 * 60 + 30 else int(1 + np.sin(t / 23.0))
            n = max(0, min(3, n))
            return self.idle_order[:n]
        n = int(round(target_ratio(t) * CAPACITY))
        n = max(0, min(CAPACITY, n + int(self.rng.integers(-2, 3))))
        return self.orders[k][:n]
