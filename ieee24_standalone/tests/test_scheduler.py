"""Behavioral checks for paired weather risk, robust risk, and all masters."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from ieee24_standalone import Master_CVAR, Master_DET, Master_DRO_CVAR, load_study
from ieee24_standalone.solver import _event_dns, _risk


ROOT = Path(__file__).parents[1]


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "bank.sqlite"
        subprocess.run([sys.executable, "-m", "ieee24_standalone.examples.make_synthetic_bank",
                        str(cls.db)], check=True, capture_output=True)
        cls.config = ROOT / "examples" / "demo_config.json"
        cls.study = load_study(cls.db, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_three_masters_use_same_bank_and_utility_floor(self):
        det = Master_DET(self.study)
        cvar = Master_CVAR(self.study, det)
        dro = Master_DRO_CVAR(self.study, det)
        for result in (det, cvar, dro):
            self.assertEqual(len(result.losses_mw), len(self.study.scenario_ids))
            self.assertEqual(set(result.starts), {t["task_id"] for t in self.study.tasks})
            self.assertGreater(result.candidates_evaluated, 0)
            self.assertGreaterEqual(result.dro_cvar_loss_mw + 1e-7, result.cvar_loss_mw)
            self.assertGreater(result.slave_calls, 0)
        self.assertGreaterEqual(cvar.utility, det.utility - self.study.settings["utility_tolerance"] - 1e-9)
        self.assertGreaterEqual(dro.utility, det.utility - self.study.settings["utility_tolerance"] - 1e-9)
        self.assertLessEqual(cvar.cvar_loss_mw, det.cvar_loss_mw + 1e-6)
        self.assertLessEqual(dro.dro_cvar_loss_mw, det.dro_cvar_loss_mw + 1e-6)

    def test_tv_zero_matches_empirical_and_large_ball_reaches_max(self):
        loss = np.array([0.0, 4.0, 10.0])
        p = np.array([0.5, 0.3, 0.2])
        _, empirical, robust_zero = _risk(loss, p, 0.5, 0.0)
        self.assertAlmostEqual(empirical, 6.4, places=7)
        self.assertAlmostEqual(robust_zero, empirical, places=7)
        _, _, robust_one = _risk(loss, p, 0.5, 1.0)
        self.assertAlmostEqual(robust_one, 10.0, places=7)

    def test_direction_changes_weather_event_weight(self):
        line = 8
        hazard = tuple({} if i != line else {
            "bearing_deg": 90, "base_rate_per_h": 0.0001,
            "speed_threshold_mps": 2, "speed_slope_per_mps": 0.5,
            "max_rate_per_h": 0.02,
        } for i in range(len(self.study.network.branch_from)))
        speed = self.study.wind_speed_mps.copy()
        speed[0, 0, line] = 20
        direction = self.study.wind_direction_deg.copy()
        direction[0, 0, line] = 90  # parallel to line
        parallel = replace(self.study, hazard=hazard, wind_speed_mps=speed, wind_direction_deg=direction)
        direction = direction.copy(); direction[0, 0, line] = 0  # transverse
        crosswind = replace(parallel, wind_direction_deg=direction)
        scores = {"base": 0.0, f"line:{line}": 100.0}
        self.assertGreater(_event_dns(crosswind, scores, 0, 0, set()),
                           _event_dns(parallel, scores, 0, 0, set()))

    def test_missing_weather_row_is_rejected(self):
        with sqlite3.connect(self.db) as con:
            con.execute("DELETE FROM weather WHERE scenario_id='calm' AND step=0 AND line_id=0")
        try:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_study(self.db, self.config)
        finally:
            with sqlite3.connect(self.db) as con:
                con.execute("INSERT INTO weather VALUES ('calm', 0, 0, 3.5, 25.0)")


if __name__ == "__main__":
    unittest.main()
