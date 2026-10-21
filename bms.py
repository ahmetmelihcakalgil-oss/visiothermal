"""Building Management System output.

MockBMS records change-of-value writes as JSON lines, using the same point
names a BACnet/IP or Modbus gateway would expose. Real adapters are stubs:
they document the intended mapping but are NOT tested against hardware.
"""
import json


class BMSAdapter:
    def write_points(self, minute, points, meta=None):
        raise NotImplementedError


# point name -> (protocol object, units)
POINT_MAP = {
    "hvac_setpoint": ("analogOutput,1", "degC"),
    "fan_speed": ("analogOutput,2", "percent"),
    "light_front": ("analogOutput,11", "percent"),
    "light_middle": ("analogOutput,12", "percent"),
    "light_back": ("analogOutput,13", "percent"),
}


class MockBMS(BMSAdapter):
    def __init__(self, path=None, device_id=1001):
        self.path = path
        self.device_id = device_id
        self.state = {}
        self.events = []

    def write_points(self, minute, points, meta=None):
        """Write only points whose value changed (change-of-value)."""
        for name, value in points.items():
            if self.state.get(name) == value:
                continue
            self.state[name] = value
            obj, unit = POINT_MAP[name]
            self.events.append({
                "minute": minute,
                "protocol": "BACnet/IP",
                "device": self.device_id,
                "object": obj,
                "point": name,
                "property": "presentValue",
                "value": value,
                "unit": unit,
                "reason": meta or {},
            })

    def flush(self):
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                for e in self.events:
                    f.write(json.dumps(e, ensure_ascii=True) + "\n")


class BACnetAdapter(BMSAdapter):
    """Intended: use BAC0 to write presentValue on POINT_MAP objects."""

    def write_points(self, minute, points, meta=None):
        raise NotImplementedError("needs a BACnet/IP gateway and site object list")


class ModbusAdapter(BMSAdapter):
    """Intended: use pymodbus to write holding registers mapped from POINT_MAP."""

    def write_points(self, minute, points, meta=None):
        raise NotImplementedError("needs a Modbus gateway and the vendor register map")
