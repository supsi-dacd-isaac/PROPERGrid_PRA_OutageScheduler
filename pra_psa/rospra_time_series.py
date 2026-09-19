"""Time-series adapter for the ROSPRA cascading-failure engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pra_psa.time_series_pra import GaussianRelativeLoadSampler


@dataclass
class TimeSeriesROSPRAResult:
    summary: dict[str, Any]
    risk_by_time: pd.DataFrame
    risk_by_contingency: pd.DataFrame
    scenario_results: pd.DataFrame

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(self.summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        self.risk_by_time.to_csv(output / "risk_by_time.csv", index=False)
        self.risk_by_contingency.to_csv(output / "risk_by_contingency.csv", index=False)
        self.scenario_results.to_csv(output / "scenario_results.csv", index=False)


def pandapower_profiles_to_bus(net: Any, load_profiles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate load-element profiles to the bus order used by ``DCNetwork``."""

    profiles = pd.DataFrame(load_profiles).copy()
    if profiles.shape[1] != len(net.load):
        raise ValueError("load_profiles must contain one column per pandapower load.")
    profiles.columns = net.load.index
    bus_profiles = pd.DataFrame(0.0, index=profiles.index, columns=net.bus.index)
    for load_index, row in net.load.iterrows():
        if bool(row.get("in_service", True)):
            bus_profiles.loc[:, row["bus"]] += profiles.loc[:, load_index].to_numpy(dtype=float)
    return bus_profiles


def run_time_series_rospra(
    engine: Any,
    baseline_bus_profiles: pd.DataFrame | np.ndarray,
    *,
    n_samples: int = 50,
    step_hours: float = 1.0,
    sampler: Any | None = None,
    seed: int = 42,
    planned_outages: tuple[str, ...] = (),
) -> TimeSeriesROSPRAResult:
    """Run ROSPRA independently at each baseline operating point.

    ROSPRA evaluates cascading consequences as expected demand not served (MWh),
    which is intentionally reported separately from the static overload/voltage
    score produced by :func:`pra_psa.time_series_pra.run_time_series_pra`.
    """

    if n_samples < 2 or step_hours <= 0:
        raise ValueError("n_samples must be at least 2 and step_hours positive.")
    profiles = pd.DataFrame(baseline_bus_profiles).copy()
    if profiles.shape[1] != engine.simulator.network.n_buses:
        raise ValueError("baseline_bus_profiles must contain one column per ROSPRA bus.")
    sampler = sampler or GaussianRelativeLoadSampler()
    rng = np.random.default_rng(seed)

    time_rows: list[dict[str, Any]] = []
    contingency_frames: list[pd.DataFrame] = []
    scenario_frames: list[pd.DataFrame] = []
    for position, (time_index, row) in enumerate(profiles.iterrows()):
        baseline = row.to_numpy(dtype=float)
        samples = sampler.sample(baseline, n_samples, rng, time_index=time_index)
        result = engine.run(
            n_samples=n_samples,
            horizon_hours=step_hours,
            load_samples=samples,
            planned_outages=planned_outages,
            random_state=seed + position,
        )
        time_rows.append(
            {
                "time": time_index,
                "expected_dns_mwh": result.summary.expected_dns_mwh,
                "probability_of_positive_dns": result.summary.probability_of_positive_dns,
                "maximum_dns_mw": result.summary.maximum_dns_mw,
                "mc_standard_error_mwh": result.summary.monte_carlo_standard_error_mwh,
                "ci95_lower_mwh": result.summary.monte_carlo_ci95_lower_mwh,
                "ci95_upper_mwh": result.summary.monte_carlo_ci95_upper_mwh,
                "mean_total_load_mw": float(samples.sum(axis=1).mean()),
            }
        )
        contingency = result.risk_by_contingency.copy()
        contingency.insert(0, "time", time_index)
        contingency_frames.append(contingency)
        scenarios = result.scenario_results.copy()
        scenarios.insert(0, "time", time_index)
        scenario_frames.append(scenarios)

    risk_by_time = pd.DataFrame(time_rows)
    risk_by_time["expected_dns_per_100mw"] = np.divide(
        100.0 * risk_by_time["expected_dns_mwh"],
        risk_by_time["mean_total_load_mw"],
        out=np.zeros(len(risk_by_time), dtype=float),
        where=risk_by_time["mean_total_load_mw"].to_numpy(dtype=float) > 0,
    )
    risk_by_contingency = pd.concat(contingency_frames, ignore_index=True)
    scenarios = pd.concat(scenario_frames, ignore_index=True)
    summary = {
        "n_operating_points": int(len(profiles)),
        "n_samples_per_operating_point": int(n_samples),
        "n_contingencies": int(len(engine.contingencies)),
        "expected_dns_mwh_sum": float(risk_by_time["expected_dns_mwh"].sum()),
        "maximum_dns_mw": float(risk_by_time["maximum_dns_mw"].max()),
        "maximum_probability_of_positive_dns": float(
            risk_by_time["probability_of_positive_dns"].max()
        ),
        "metric": "probability-weighted demand not served from sequential DC cascading simulation",
    }
    return TimeSeriesROSPRAResult(summary, risk_by_time, risk_by_contingency, scenarios)
