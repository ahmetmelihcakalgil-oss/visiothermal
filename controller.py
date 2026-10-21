"""Edge decision layer: occupancy -> HVAC setpoint, fan speed, lighting.

Rules that keep it safe for comfort:
  * stepping UP in occupancy level is immediate
  * stepping DOWN needs `down_minutes` consecutive low readings plus a
    hysteresis margin, so short breaks do not make the system flap
  * if the camera/detector fails, fall back to the normal 100% schedule
"""
from dataclasses import dataclass, field

LEVELS = (0, 25, 50, 100)

# Occupancy ratio edges between levels: [0-5%) -> 0, [5-35%) -> 25, [35-70%) -> 50, else 100
EDGES = (0.05, 0.35, 0.70)

# Heating-season setpoints in deg C (setback when empty)
SETPOINT = {0: 17.0, 25: 19.5, 50: 21.0, 100: 22.0}
FAN_PCT = {0: 10, 25: 35, 50: 65, 100: 100}
MAX_CFM = 4000

# Share of full HVAC power drawn at each level. ASSUMPTION derived from the
# PDR's 60-80% idle reduction target; must be calibrated with site data.
POWER_FRACTION = {0: 0.20, 25: 0.45, 50: 0.70, 100: 1.00}

LIGHT_ACTIVE = 100
LIGHT_IDLE = 15   # never fully dark: egress minimum


def level_for(ratio, shift=0.0):
    lv = 0
    for edge, level in zip(EDGES, LEVELS[1:]):
        if ratio >= edge - shift:
            lv = level
    return lv


@dataclass
class Decision:
    level: int
    setpoint_c: float
    fan_pct: int
    fan_cfm: int
    lights_pct: list = field(default_factory=list)
    fallback: bool = False
    ratio: float = 0.0


class Controller:
    def __init__(self, capacity, down_minutes=5, hysteresis=0.03, light_hold=5, bands=3):
        self.capacity = capacity
        self.down_minutes = down_minutes
        self.hysteresis = hysteresis
        self.light_hold = light_hold
        self.level = 100          # start in the safe state
        self._down = 0
        self._light_empty = [0] * bands
        self._light = [LIGHT_ACTIVE] * bands

    def step(self, count, band_counts, sensor_ok=True):
        """One control step (one minute in the demo)."""
        if not sensor_ok or count is None:
            self.level, self._down = 100, 0
            self._light = [LIGHT_ACTIVE] * len(self._light)
            return self._decision(0.0, fallback=True)

        ratio = count / float(self.capacity)
        up = level_for(ratio)
        down = level_for(ratio, shift=self.hysteresis)
        if up > self.level:
            self.level, self._down = up, 0
        elif down < self.level:
            self._down += 1
            if self._down >= self.down_minutes:
                self.level, self._down = down, 0
        else:
            self._down = 0

        for i, n in enumerate(band_counts):
            if n > 0:
                self._light_empty[i] = 0
                self._light[i] = LIGHT_ACTIVE
            else:
                self._light_empty[i] += 1
                if self._light_empty[i] >= self.light_hold:
                    self._light[i] = LIGHT_IDLE
        return self._decision(ratio)

    def _decision(self, ratio, fallback=False):
        return Decision(
            level=self.level,
            setpoint_c=SETPOINT[self.level],
            fan_pct=FAN_PCT[self.level],
            fan_cfm=int(MAX_CFM * FAN_PCT[self.level] / 100),
            lights_pct=list(self._light),
            fallback=fallback,
            ratio=ratio,
        )
