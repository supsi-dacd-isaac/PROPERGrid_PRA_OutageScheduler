"""Create a tiny fixed *illustrative* bank; no measured wind or calibrated risk."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from ieee24_standalone.model import _read_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bank: {args.output}")
    net = _read_case(Path(__file__).parents[1] / "case" / "System.m")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scenarios = (("calm", 1 / 3, 0.99, 5.0, 25.0),
                 ("crosswind", 1 / 3, 1.04, 17.0, 120.0),
                 ("peak", 1 / 3, 1.10, 24.0, 145.0))
    with sqlite3.connect(args.output) as con:
        con.executescript("""
            CREATE TABLE scenario(scenario_id TEXT PRIMARY KEY, probability REAL NOT NULL);
            CREATE TABLE time_step(step INTEGER PRIMARY KEY, timestamp_utc TEXT UNIQUE NOT NULL, duration_h REAL NOT NULL);
            CREATE TABLE demand(scenario_id TEXT, step INTEGER, bus_id INTEGER, demand_mw REAL NOT NULL,
                                PRIMARY KEY(scenario_id, step, bus_id));
            CREATE TABLE weather(scenario_id TEXT, step INTEGER, line_id INTEGER, speed_mps REAL NOT NULL,
                                 direction_deg REAL NOT NULL, PRIMARY KEY(scenario_id, step, line_id));
        """)
        con.executemany("INSERT INTO scenario VALUES (?, ?)", [(s, p) for s, p, *_ in scenarios])
        con.executemany("INSERT INTO time_step VALUES (?, ?, ?)", [
            (t, f"2026-01-0{t + 1}T00:00:00Z", 24.0) for t in range(3)])
        demand_rows, weather_rows = [], []
        for s, _, multiplier, wind, direction in scenarios:
            for t in range(3):
                day_factor = (0.98, 1.03, 1.00)[t]
                demand_rows.extend((s, t, int(bus), float(mw * multiplier * day_factor))
                                   for bus, mw in zip(net.bus_ids, net.demand_base))
                weather_rows.extend((s, t, line, float(wind + (t - 1) * 1.5),
                                     float((direction + 2 * line) % 360))
                                    for line in range(len(net.branch_from)))
        con.executemany("INSERT INTO demand VALUES (?, ?, ?, ?)", demand_rows)
        con.executemany("INSERT INTO weather VALUES (?, ?, ?, ?, ?)", weather_rows)
    print(f"Wrote illustrative bank: {args.output} ({len(scenarios)} paths, 3 steps)")


if __name__ == "__main__":
    main()
