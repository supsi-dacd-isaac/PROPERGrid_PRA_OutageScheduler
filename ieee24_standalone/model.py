"""MATPOWER case and strictly validated, fixed SQLite scenario bank."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Network:
    base_mva: float
    bus_ids: np.ndarray
    demand_base: np.ndarray
    gen_bus: np.ndarray
    gen_pmax: np.ndarray
    branch_from: np.ndarray
    branch_to: np.ndarray
    branch_x: np.ndarray
    branch_rate: np.ndarray
    branch_tap: np.ndarray
    branch_shift: np.ndarray


@dataclass(frozen=True)
class StudyData:
    network: Network
    scenario_ids: tuple[str, ...]
    probabilities: np.ndarray
    timestamps: tuple[str, ...]
    duration_h: np.ndarray
    demand_mw: np.ndarray  # scenario, step, bus
    wind_speed_mps: np.ndarray  # scenario, step, branch
    wind_direction_deg: np.ndarray  # scenario, step, branch
    tasks: tuple[dict, ...]
    hazard: tuple[dict, ...]
    settings: dict


def _read_case(path: Path) -> Network:
    source = path.read_text(encoding="utf-8")

    def matrix(name: str) -> np.ndarray:
        match = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", source, re.S)
        if not match:
            raise ValueError(f"Missing MATPOWER matrix: {name}")
        rows = []
        for line in match.group(1).splitlines():
            line = line.split("%", 1)[0].strip().rstrip(";")
            if line:
                rows.append([float(value) for value in line.split()])
        return np.asarray(rows, dtype=float)

    bus, gen, branch = matrix("bus"), matrix("gen"), matrix("branch")
    base = re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)\s*;", source)
    if base is None or len(set(bus[:, 0].astype(int))) != len(bus):
        raise ValueError("Invalid MATPOWER bus or baseMVA data")
    bus_ids = bus[:, 0].astype(int)
    lookup = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    if not set(gen[:, 0].astype(int)).issubset(lookup) or not set(branch[:, :2].astype(int).flat).issubset(lookup):
        raise ValueError("A generator or branch references an unknown bus")
    if np.any(branch[:, 3] <= 0) or np.any(branch[:, 10] != 1):
        raise ValueError("The standalone DC model requires active branches with positive reactance")
    return Network(
        float(base.group(1)), bus_ids, bus[:, 2],
        np.asarray([lookup[int(i)] for i in gen[:, 0]], dtype=int),
        np.where(gen[:, 7] > 0, gen[:, 8], 0.0),
        np.asarray([lookup[int(i)] for i in branch[:, 0]], dtype=int),
        np.asarray([lookup[int(i)] for i in branch[:, 1]], dtype=int),
        branch[:, 3], np.where(branch[:, 5] > 0, branch[:, 5], 1e6),
        np.where(branch[:, 8] != 0, branch[:, 8], 1.0),
        np.deg2rad(branch[:, 9]),
    )


def load_study(scenario_db: str | Path, config_json: str | Path, case_file: str | Path | None = None) -> StudyData:
    """Read all inputs once; the optimizers never sample or mutate the bank."""
    network = _read_case(Path(case_file or Path(__file__).parent / "case" / "System.m"))
    config = json.loads(Path(config_json).read_text(encoding="utf-8"))
    nb, ng, nl = len(network.bus_ids), len(network.gen_bus), len(network.branch_from)
    tasks = tuple(config["tasks"])
    if not tasks or len({str(t["task_id"]) for t in tasks}) != len(tasks):
        raise ValueError("Tasks must be nonempty and have unique task_id values")
    with sqlite3.connect(str(scenario_db)) as con:
        sc = con.execute("SELECT scenario_id, probability FROM scenario ORDER BY scenario_id").fetchall()
        ti = con.execute("SELECT step, timestamp_utc, duration_h FROM time_step ORDER BY step").fetchall()
        if not sc or not ti or [int(r[0]) for r in ti] != list(range(len(ti))):
            raise ValueError("Scenario and consecutive time_step records are required")
        sid = tuple(str(r[0]) for r in sc)
        prob = np.asarray([r[1] for r in sc], dtype=float)
        durations = np.asarray([r[2] for r in ti], dtype=float)
        stamps = tuple(str(r[1]) for r in ti)
        dates = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps]
        if any(d.tzinfo is None or d.utcoffset().total_seconds() != 0 for d in dates) or dates != sorted(set(dates)):
            raise ValueError("time_step timestamps must be distinct, increasing UTC instants")
        if not np.all(np.isfinite(prob)) or np.any(prob < 0) or not np.isclose(prob.sum(), 1, atol=1e-9):
            raise ValueError("Scenario probabilities must be nonnegative and sum to one")
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("Each time step needs a positive duration_h")
        shape = (len(sc), len(ti))
        demand = np.full((*shape, nb), np.nan)
        speed = np.full((*shape, nl), np.nan)
        direction = np.full((*shape, nl), np.nan)
        si = {s: i for i, s in enumerate(sid)}
        bi = {int(b): i for i, b in enumerate(network.bus_ids)}
        seen_d, seen_w = set(), set()
        for s, t, b, mw in con.execute("SELECT scenario_id, step, bus_id, demand_mw FROM demand"):
            key = (s, t, b)
            if s not in si or t not in range(shape[1]) or b not in bi or key in seen_d:
                raise ValueError(f"Invalid or duplicate demand key: {key}")
            seen_d.add(key)
            demand[si[s], t, bi[b]] = mw
        for s, t, line, v, phi in con.execute("SELECT scenario_id, step, line_id, speed_mps, direction_deg FROM weather"):
            key = (s, t, line)
            if s not in si or t not in range(shape[1]) or not 0 <= line < nl or key in seen_w:
                raise ValueError(f"Invalid or duplicate weather key: {key}")
            seen_w.add(key)
            speed[si[s], t, line] = v
            direction[si[s], t, line] = phi
    if any(np.any(~np.isfinite(a)) for a in (demand, speed, direction)):
        raise ValueError("Incomplete or nonfinite scenario × step × bus/branch grid")
    if np.any(demand < 0) or np.any(speed < 0) or np.any((direction < 0) | (direction >= 360)):
        raise ValueError("Demand and wind speed must be nonnegative; direction must be in [0, 360)")

    defaults = dict(alpha=0.95, tv_radius=0.1, max_active=2, max_candidates=8,
                    utility_tolerance=0.0, redispatch_fraction=1.0,
                    contingency_lines=list(range(nl)), contingency_generators=[],
                    loss_scale=1.0, proxy_gain=0.1,
                    representatives_per_topology=0)
    settings = defaults | config.get("settings", {})
    h = tuple(config.get("hazard", {}).get(str(line), {}) for line in range(nl))
    for line, spec in enumerate(h):
        if any(float(spec.get(k, 0)) < 0 for k in ("base_rate_per_h", "speed_threshold_mps", "speed_slope_per_mps", "max_rate_per_h")):
            raise ValueError(f"Negative hazard parameter for branch {line}")
        if spec and not 0 <= float(spec.get("bearing_deg", 0)) < 180:
            raise ValueError(f"Branch {line} bearing must be in [0, 180) degrees")
        if float(spec.get("base_rate_per_h", 0.0)) > 0 and line not in settings["contingency_lines"]:
            raise ValueError(f"Hazard branch {line} is absent from contingency_lines")
    for task in tasks:
        typ, asset = task["asset_type"], int(task["asset_id"])
        if typ not in ("line", "gen") or not 0 <= asset < (nl if typ == "line" else ng):
            raise ValueError(f"Unknown asset in task {task['task_id']}")
        d = int(task["duration_steps"])
        lo, hi = int(task["earliest_start"]), int(task["latest_start"])
        if d < 1 or not 0 <= lo <= hi or hi + d > shape[1]:
            raise ValueError(f"Invalid window/duration for task {task['task_id']}")
        if not np.isfinite(float(task["utility"])):
            raise ValueError("Task utility must be finite")
    if len({(x["asset_type"], int(x["asset_id"])) for x in tasks}) != len(tasks):
        raise ValueError("Only one maintenance task per physical asset is supported")
    if not 0 < settings["alpha"] < 1 or not 0 <= settings["tv_radius"] <= 1:
        raise ValueError("alpha must be in (0,1) and TV radius in [0,1]")
    if settings["max_active"] < 1 or settings["max_candidates"] < 1:
        raise ValueError("Positive max_active and max_candidates are required")
    if int(settings["representatives_per_topology"]) != settings["representatives_per_topology"] or settings["representatives_per_topology"] < 0:
        raise ValueError("representatives_per_topology must be a nonnegative integer")
    for key, upper in (("contingency_lines", nl), ("contingency_generators", ng)):
        if any(int(i) != i or not 0 <= i < upper for i in settings[key]):
            raise ValueError(f"Invalid {key} index")
    return StudyData(network, sid, prob, stamps, durations, demand, speed, direction, tasks, h, settings)
