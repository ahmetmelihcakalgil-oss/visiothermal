import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import synth
import detector as detmod
import energy
from controller import Controller
from bms import MockBMS
from pipeline import Pipeline

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def make_detector():
    det = detmod.BlobDetector()
    det.calibrate(synth.empty_frame(),
                  synth.render([5 * synth.COLS + 4], np.random.default_rng(0), noise=0))
    return det


class DetectorTests(unittest.TestCase):
    def test_counts_close_to_truth(self):
        det = make_detector()
        rng = np.random.default_rng(11)
        order = np.random.default_rng(5).permutation(synth.CAPACITY)
        for n in (0, 5, 30, 60, 90):
            errs = []
            for _ in range(6):
                f = synth.render(order[:n], rng, gain=0.92 + 0.16 * rng.random())
                errs.append(abs(len(det.detect(f)) - n))
            self.assertLessEqual(np.mean(errs), max(1.0, 0.04 * n), "n=%d" % n)

    def test_empty_room_has_no_false_people(self):
        det = make_detector()
        rng = np.random.default_rng(2)
        for _ in range(10):
            f = synth.render([], rng, gain=0.9 + 0.2 * rng.random())
            self.assertEqual(len(det.detect(f)), 0)


class ControllerTests(unittest.TestCase):
    def test_step_up_is_immediate_and_down_is_delayed(self):
        c = Controller(96, down_minutes=5)
        self.assertEqual(c.step(80, [1, 1, 1]).level, 100)
        for _ in range(4):
            self.assertEqual(c.step(0, [0, 0, 0]).level, 100)
        self.assertEqual(c.step(0, [0, 0, 0]).level, 0)
        self.assertEqual(c.step(60, [1, 1, 1]).level, 50)

    def test_no_flapping_near_an_edge(self):
        c = Controller(96)
        c.step(10, [1, 0, 0])
        for _ in range(5):
            c.step(10, [1, 0, 0])
        changes, last = 0, c.level
        for k in range(40):
            lvl = c.step(34 if k % 2 else 36, [1, 1, 1]).level   # ratio 0.354 / 0.375
            if lvl != last:
                changes += 1
                last = lvl
        self.assertLessEqual(changes, 1)

    def test_failsafe_goes_to_full_schedule(self):
        c = Controller(96, down_minutes=1)
        for _ in range(3):
            c.step(0, [0, 0, 0])
        self.assertEqual(c.level, 0)
        d = c.step(None, [0, 0, 0], sensor_ok=False)
        self.assertEqual((d.level, d.fallback), (100, True))
        self.assertEqual(d.lights_pct, [100, 100, 100])

    def test_empty_band_dims_lights_after_hold(self):
        c = Controller(96, light_hold=3)
        for _ in range(3):
            d = c.step(20, [5, 0, 0])
        self.assertEqual(d.lights_pct[0], 100)
        self.assertLess(d.lights_pct[2], 100)


class EnergyAndBmsTests(unittest.TestCase):
    def test_baseline_matches_pdr(self):
        p = energy.Plant()
        kwh_day = p.hvac_kw * p.hours_per_day
        self.assertEqual(kwh_day, 180.0)
        idle_waste_year = 4 * p.hvac_kw * p.academic_days      # 4 h/day at 15 kW
        self.assertEqual(idle_waste_year, 12000.0)
        self.assertEqual(idle_waste_year * p.halls, 600000.0)

    def test_bms_writes_only_on_change(self):
        b = MockBMS()
        pts = {"hvac_setpoint": 22.0, "fan_speed": 100,
               "light_front": 100, "light_middle": 100, "light_back": 100}
        b.write_points(0, pts)
        n = len(b.events)
        b.write_points(1, dict(pts))
        self.assertEqual(len(b.events), n)
        b.write_points(2, dict(pts, fan_speed=35))
        self.assertEqual(len(b.events), n + 1)


class PrivacyTests(unittest.TestCase):
    def test_pipeline_writes_no_files(self):
        pipe = Pipeline(make_detector())
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                for m in range(8):
                    pipe.process(synth.render(range(20), rng), m)
                pipe.process(None, 9)
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)

    def test_runtime_modules_never_save_images(self):
        for name in ("detector.py", "pipeline.py", "controller.py", "bms.py", "live.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
                src = fh.read()
            for bad in ("imwrite", "imencode", "VideoWriter", ".save("):
                self.assertNotIn(bad, src, "%s uses %s" % (name, bad))


class RealSetTests(unittest.TestCase):
    def test_missing_dataset_is_skipped(self):
        import realset
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(realset, "DATASET_DIR", tmp):
                out = realset.analyze("blob")
        self.assertEqual(out["items"], [])
        self.assertIsNone(out["detector"])

    def test_stills_produce_dashboard_items(self):
        import realset
        out = realset.analyze("blob")          # offline detector, no YOLO needed
        self.assertEqual(out["detector"], "blob")
        self.assertEqual(len(out["items"]), len(realset.FRAMES))
        for it in out["items"]:
            self.assertGreaterEqual(it["count"], 0)
            self.assertIn(it["level"], (0, 25, 50, 100))
            self.assertEqual(len(it["lights"]), 3)
            self.assertTrue(it["jpg"])
        # the hand-checked count for 11:24 must override the detector
        by_file = {it["file"]: it for it in out["items"]}
        self.assertEqual(by_file["frame_1124.jpg"]["level"], 0)   # blob finds nobody here

    def test_merge_boxes_dedups_and_caps(self):
        import realset
        a = [(0, 0, 10, 10, 0.9), (20, 0, 30, 10, 0.8)]
        b = [(0, 0, 10, 10, 0.6), (40, 0, 50, 10, 0.2)]
        self.assertEqual(len(realset.merge_boxes(a, b, limit=10)), 3)
        self.assertEqual(len(realset.merge_boxes(a, b, limit=2)), 2)

    def test_top_band_is_the_back_of_the_class(self):
        import realset
        ranges = realset.band_ranges(1000)
        boxes = [(0, ranges[0][0], 10, 20)]       # one person in the far/top rows
        _, _, _, _, lights, bands = realset.decision(10, boxes, ranges)
        self.assertEqual(bands, [0, 0, 1])        # ön, orta, arka
        self.assertEqual(lights, [15, 15, 100])   # front+middle off, back on


if __name__ == "__main__":
    unittest.main()
