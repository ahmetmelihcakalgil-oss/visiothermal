"""Frame -> count -> decision -> BMS. Frames live only in memory."""
from collections import deque

import detector as det
from controller import Controller
from bms import MockBMS
import synth


class Pipeline:
    def __init__(self, detector, bms=None, capacity=synth.CAPACITY, smooth=3):
        self.detector = detector
        self.bms = bms or MockBMS()
        self.controller = Controller(capacity)
        self._recent = deque(maxlen=smooth)
        self.last_count = None

    def process(self, frame, minute, width=synth.W, height=synth.H):
        """Handle one frame. The frame is never written anywhere.

        frame=None models a camera or detector failure (fail-safe path).
        """
        if frame is None:
            decision = self.controller.step(None, [0, 0, 0], sensor_ok=False)
            boxes, raw = [], None
            self.last_count = None
        else:
            boxes = self.detector.detect(frame)
            raw = len(boxes)
            self._recent.append(raw)
            count = int(sorted(self._recent)[len(self._recent) // 2])  # median filter
            self.last_count = count
            bands = det.band_counts(boxes, height, 40, height - 20)
            decision = self.controller.step(count, bands)

        la = decision.lights_pct
        self.bms.write_points(minute, {
            "hvac_setpoint": decision.setpoint_c,
            "fan_speed": decision.fan_pct,
            "light_front": la[0],
            "light_middle": la[1],
            "light_back": la[2],
        }, meta={"level": decision.level, "ratio": round(decision.ratio, 3),
                 "fallback": decision.fallback})
        return decision, boxes, raw
