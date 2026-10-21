"""Energy accounting: baseline schedule vs occupancy-driven operation.

Numbers follow the PDR where it gives them (15 kW HVAC per hall, 12 h/day,
200 academic days, 50 halls). Lighting power is an extra ASSUMPTION.
"""
from dataclasses import dataclass

from controller import POWER_FRACTION, LIGHT_ACTIVE


@dataclass(frozen=True)
class Plant:
    hvac_kw: float = 15.0
    lighting_kw: float = 2.0        # assumption, not in the PDR
    hours_per_day: int = 12
    academic_days: int = 200
    halls: int = 50


def baseline_kw(plant):
    return plant.hvac_kw, plant.lighting_kw


def optimized_kw(plant, level, lights_pct):
    hvac = plant.hvac_kw * POWER_FRACTION[level]
    light = plant.lighting_kw * (sum(lights_pct) / len(lights_pct)) / float(LIGHT_ACTIVE)
    return hvac, light


def summarize(plant, hvac_base_kwh, hvac_opt_kwh, light_base_kwh, light_opt_kwh):
    def pct(base, opt):
        return 100.0 * (base - opt) / base if base else 0.0

    days = plant.academic_days
    return {
        "hvac_base_kwh_day": hvac_base_kwh,
        "hvac_opt_kwh_day": hvac_opt_kwh,
        "hvac_saving_pct": pct(hvac_base_kwh, hvac_opt_kwh),
        "light_base_kwh_day": light_base_kwh,
        "light_opt_kwh_day": light_opt_kwh,
        "light_saving_pct": pct(light_base_kwh, light_opt_kwh),
        "total_saving_pct": pct(hvac_base_kwh + light_base_kwh, hvac_opt_kwh + light_opt_kwh),
        "saved_kwh_year_per_hall": (hvac_base_kwh + light_base_kwh - hvac_opt_kwh - light_opt_kwh) * days,
        "saved_kwh_year_campus": (hvac_base_kwh + light_base_kwh - hvac_opt_kwh - light_opt_kwh) * days * plant.halls,
        "campus_halls": plant.halls,
        "academic_days": days,
    }
