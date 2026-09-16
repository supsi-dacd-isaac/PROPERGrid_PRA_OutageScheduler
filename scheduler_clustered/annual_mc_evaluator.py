"""Paired full-horizon Monte Carlo N-1 validation of two outage schedules.

The evaluator uses common random numbers: each Monte Carlo annual trajectory is
identical for the deterministic and CVaR schedules. At every time step it solves
normal operation and all valid N-1 states through :class:`ClusterSCOPFOracle`.

Two risk quantities are retained:

* raw worst-contingency DNS for the scheduled topology;
* positive incremental DNS relative to a no-maintenance topology solved at the
  same stochastic demand and with the same effective contingency set.

Without contingency occurrence probabilities, the annual sums are DNS exposure
metrics, not formal EENS. If the planning time step is one hour, their units are
MWh.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scheduler_clustered.risk_metrics import weighted_var_cvar
from scheduler_clustered.scenario_sampling import demand_fingerprint
from scheduler_clustered.serialization import json_safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnnualMCConfig:
    mc_years: int = 20
    seed: int = 20260731
    alpha: float = 0.95
    temporal_rho: float = 0.90
    relative_sigma: float = 0.05
    global_sigma: float = 0.025
    correlation_shrinkage: float = 0.20
    antithetic: bool = True
    truncate_multiplier_min: float = 0.60
    truncate_multiplier_max: float = 1.50
    time_step_hours: float = 1.0
    oracle_workers: int = 1
    total_dns_component_weight: float = 0.10
    use_all_contingencies: bool = True
    checkpoint_every_years: int = 1
    winner_tolerance_mwh: float = 1e-6


@dataclass(frozen=True)
class HourlyMCRecord:
    mc_year: int
    formulation: str
    time_index: int
    time: str
    total_demand_mw: float
    active_outages: tuple[str, ...]
    effective_contingency_count: int
    candidate_worst_dns_mw: float
    candidate_total_dns_mw: float
    candidate_worst_contingency: str | None
    baseline_worst_dns_mw: float
    baseline_total_dns_mw: float
    incremental_worst_dns_mw: float
    incremental_total_dns_mw: float
    composite_incremental_loss_mw: float


class FullHorizonTrajectorySampler:
    """Generate spatially and temporally correlated Gaussian load trajectories."""

    def __init__(self, base_demand: np.ndarray, config: AnnualMCConfig):
        demand = np.asarray(base_demand, dtype=float)
        if demand.ndim != 2:
            raise ValueError("base_demand must have shape (time, bus).")
        self.base = demand
        self.config = config
        positive = demand > 1e-9
        normalized = np.zeros_like(demand)
        means = np.divide(demand.sum(axis=0), positive.sum(axis=0), out=np.ones(demand.shape[1]), where=positive.sum(axis=0) > 0)
        normalized = np.divide(demand, means, out=np.zeros_like(demand), where=means > 1e-9)
        if demand.shape[0] > 2 and demand.shape[1] > 1:
            corr = np.corrcoef(normalized, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            corr = np.eye(demand.shape[1])
        shrink = float(np.clip(config.correlation_shrinkage, 0.0, 1.0))
        corr = (1.0 - shrink) * corr + shrink * np.eye(demand.shape[1])
        corr = 0.5 * (corr + corr.T)
        eigval, eigvec = np.linalg.eigh(corr)
        eigval = np.maximum(eigval, 1e-8)
        self.cholesky = eigvec @ np.diag(np.sqrt(eigval))

    def trajectory(self, mc_year: int) -> np.ndarray:
        # Antithetic pairing reduces comparison noise while preserving pairing.
        pair_index = mc_year // 2 if self.config.antithetic else mc_year
        sign = -1.0 if self.config.antithetic and mc_year % 2 else 1.0
        rng = np.random.default_rng(self.config.seed + pair_index)
        t_count, b_count = self.base.shape
        rho = float(np.clip(self.config.temporal_rho, -0.999, 0.999))
        innovation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        latent = np.zeros((t_count, b_count), dtype=float)
        state = rng.standard_normal(b_count) @ self.cholesky.T
        global_state = float(rng.standard_normal())
        for t in range(t_count):
            spatial_eps = rng.standard_normal(b_count) @ self.cholesky.T
            global_eps = float(rng.standard_normal())
            state = rho * state + innovation_scale * spatial_eps
            global_state = rho * global_state + innovation_scale * global_eps
            latent[t] = sign * (
                self.config.relative_sigma * state
                + self.config.global_sigma * global_state
            )
        multiplier = np.exp(latent - 0.5 * np.var(latent, axis=0, keepdims=True))
        multiplier = np.clip(
            multiplier,
            self.config.truncate_multiplier_min,
            self.config.truncate_multiplier_max,
        )
        sampled = np.maximum(0.0, self.base * multiplier)
        return sampled


def _extract_schedule(document: Mapping[str, Any], label: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """Extract schedule from standalone or common-comparison result documents."""
    aliases = {
        "deterministic": ("deterministic", "risk_neutral"),
        "cvar": ("cvar",),
    }
    for key in aliases[label]:
        section = document.get(key)
        if isinstance(section, Mapping):
            schedule = section.get("best_schedule") or {}
            if schedule:
                return dict(schedule), tuple(section.get("deferred_outages") or ())
    schedule = document.get("best_schedule") or {}
    if schedule:
        return dict(schedule), tuple(document.get("deferred_outages") or ())
    raise ValueError(f"Could not find a non-empty {label} schedule in result document.")


def load_schedule_pair(
    deterministic_path: str | Path,
    cvar_path: str | Path | None = None,
) -> dict[str, tuple[dict[str, str], tuple[str, ...]]]:
    first = json.loads(Path(deterministic_path).read_text(encoding="utf-8"))
    if cvar_path is None:
        second = first
    else:
        second = json.loads(Path(cvar_path).read_text(encoding="utf-8"))
    return {
        "deterministic": _extract_schedule(first, "deterministic"),
        "cvar": _extract_schedule(second, "cvar"),
    }


def _expanded_active(data: Mapping[str, Any], schedule: Mapping[str, str], deferred: Sequence[str]) -> dict[str, tuple[str, ...]]:
    from .master_problem import build_active_outages

    expected = set(data["names"]["outages"]) - set(deferred)
    missing = expected - set(schedule)
    if missing:
        raise ValueError("Schedule misses outage starts for: " + ", ".join(sorted(missing)))
    return build_active_outages(data, schedule)


def _annual_metrics(values: np.ndarray, alpha: float) -> dict[str, float]:
    mapping = {f"year_{i:04d}": float(v) for i, v in enumerate(values)}
    probabilities = {key: 1.0 / max(1, len(mapping)) for key in mapping}
    risk = weighted_var_cvar(mapping, probabilities, alpha=alpha)
    return {
        "mean": float(risk.expected_loss),
        "var": float(risk.var),
        "cvar": float(risk.cvar),
        "maximum": float(np.max(values) if values.size else 0.0),
        "minimum": float(np.min(values) if values.size else 0.0),
        "probability_positive": float(np.mean(values > 1e-9) if values.size else 0.0),
    }


def evaluate_full_horizon_mc(
    data: Mapping[str, Any],
    schedule_pair: Mapping[str, tuple[Mapping[str, str], Sequence[str]]],
    config: AnnualMCConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    from .cluster_scopf_oracle import ClusterSCOPFOracle
    from .annual_mc_visualization import plot_annual_mc_results

    out = Path(output_dir)
    results_dir = out / "results"
    visuals_dir = out / "visuals"
    results_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    times = tuple(str(x) for x in data["T"])
    base_demand = data["nodal_demand"].to_numpy(dtype=float)
    if len(times) != base_demand.shape[0]:
        raise ValueError("Time labels and nodal demand rows are inconsistent.")

    active = {
        label: _expanded_active(data, pair[0], pair[1])
        for label, pair in schedule_pair.items()
    }
    sampler = FullHorizonTrajectorySampler(base_demand, config)
    oracle = ClusterSCOPFOracle(data)
    contingencies = tuple(data["names"].get("contingencies", ()))
    if not config.use_all_contingencies:
        contingencies = tuple(data.get("screened_contingencies", contingencies))

    baseline_cache: dict[tuple[str, tuple[str, ...]], Any] = {}
    cache_lock = threading.Lock()

    def solve_hour(mc_year: int, formulation: str, t_idx: int, demand: np.ndarray) -> HourlyMCRecord:
        time = times[t_idx]
        active_outages = tuple(active[formulation][time])
        effective = oracle.effective_contingencies(active_outages, contingencies)
        candidate = oracle.solve(time, active_outages, contingencies=effective, demand_override=demand)
        if not math.isfinite(candidate.maximum_dns):
            raise RuntimeError(f"SCOPF failed for MC year {mc_year}, {formulation}, {time}.")

        cache_key = (demand_fingerprint(demand), tuple(effective))
        with cache_lock:
            baseline = baseline_cache.get(cache_key)
        if baseline is None:
            solved = oracle.solve(time, (), contingencies=effective, demand_override=demand)
            if not math.isfinite(solved.maximum_dns):
                raise RuntimeError(f"Baseline SCOPF failed for MC year {mc_year}, {time}.")
            with cache_lock:
                baseline = baseline_cache.setdefault(cache_key, solved)

        inc_max = max(0.0, float(candidate.maximum_dns - baseline.maximum_dns))
        inc_total = max(0.0, float(candidate.total_dns - baseline.total_dns))
        mean_inc = inc_total / max(1, len(candidate.contingencies) + 1)
        composite = inc_max + config.total_dns_component_weight * mean_inc
        return HourlyMCRecord(
            mc_year=mc_year,
            formulation=formulation,
            time_index=t_idx,
            time=time,
            total_demand_mw=float(np.sum(demand)),
            active_outages=active_outages,
            effective_contingency_count=len(effective),
            candidate_worst_dns_mw=float(candidate.maximum_dns),
            candidate_total_dns_mw=float(candidate.total_dns),
            candidate_worst_contingency=candidate.worst_contingency,
            baseline_worst_dns_mw=float(baseline.maximum_dns),
            baseline_total_dns_mw=float(baseline.total_dns),
            incremental_worst_dns_mw=inc_max,
            incremental_total_dns_mw=inc_total,
            composite_incremental_loss_mw=float(composite),
        )

    all_records: list[HourlyMCRecord] = []
    annual_rows: list[dict[str, Any]] = []
    workers = max(1, int(config.oracle_workers))

    for mc_year in range(config.mc_years):
        logger.info("Starting full-horizon MC year %d/%d", mc_year + 1, config.mc_years)
        trajectory = sampler.trajectory(mc_year)
        tasks = [(label, t_idx) for t_idx in range(len(times)) for label in ("deterministic", "cvar")]
        year_records: list[HourlyMCRecord] = []
        if workers == 1:
            for task_idx, (label, t_idx) in enumerate(tasks, start=1):
                logger.info("MC year %d: state %d/%d %s %s", mc_year, task_idx, len(tasks), label, times[t_idx])
                year_records.append(solve_hour(mc_year, label, t_idx, trajectory[t_idx]))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(solve_hour, mc_year, label, t_idx, trajectory[t_idx]): (label, t_idx)
                    for label, t_idx in tasks
                }
                for future in as_completed(futures):
                    year_records.append(future.result())

        year_records.sort(key=lambda x: (x.time_index, x.formulation))
        all_records.extend(year_records)
        for label in ("deterministic", "cvar"):
            selected = [r for r in year_records if r.formulation == label]
            dt = float(config.time_step_hours)
            annual_rows.append({
                "mc_year": mc_year,
                "formulation": label,
                "annual_raw_worst_dns_mwh": float(sum(r.candidate_worst_dns_mw for r in selected) * dt),
                "annual_incremental_worst_dns_mwh": float(sum(r.incremental_worst_dns_mw for r in selected) * dt),
                "annual_incremental_composite_loss_mwh": float(sum(r.composite_incremental_loss_mw for r in selected) * dt),
                "maximum_hourly_incremental_dns_mw": float(max((r.incremental_worst_dns_mw for r in selected), default=0.0)),
                "hours_positive_incremental_dns": float(sum(r.incremental_worst_dns_mw > 1e-9 for r in selected) * dt),
            })

        if (mc_year + 1) % max(1, config.checkpoint_every_years) == 0:
            pd.DataFrame([asdict(r) for r in all_records]).to_csv(results_dir / "hourly_mc_records_checkpoint.csv", index=False)
            pd.DataFrame(annual_rows).to_csv(results_dir / "annual_mc_summary_checkpoint.csv", index=False)

    hourly_df = pd.DataFrame([asdict(r) for r in all_records])
    annual_df = pd.DataFrame(annual_rows)
    hourly_df.to_csv(results_dir / "hourly_mc_records.csv", index=False)
    annual_df.to_csv(results_dir / "annual_mc_summary.csv", index=False)

    risk_metrics: dict[str, Any] = {}
    for label in ("deterministic", "cvar"):
        values = annual_df.loc[annual_df.formulation == label, "annual_incremental_worst_dns_mwh"].to_numpy(float)
        risk_metrics[label] = _annual_metrics(values, config.alpha)

    pivot = annual_df.pivot(index="mc_year", columns="formulation", values="annual_incremental_worst_dns_mwh")
    differences = (pivot["cvar"] - pivot["deterministic"]).to_numpy(float)
    cvar_diff = risk_metrics["cvar"]["cvar"] - risk_metrics["deterministic"]["cvar"]
    tol = config.winner_tolerance_mwh
    winner = "cvar" if cvar_diff < -tol else "deterministic" if cvar_diff > tol else "tie"

    summary = {
        "method": "paired_full_horizon_monte_carlo_n_minus_1",
        "interpretation_note": "Annual sums are DNS exposure metrics. They are EENS only when the time step is hourly and contingency probabilities are explicitly represented.",
        "config": asdict(config),
        "horizon_steps": len(times),
        "contingency_count": len(contingencies),
        "same_schedule": schedule_pair["deterministic"][0] == schedule_pair["cvar"][0],
        "risk_metrics": risk_metrics,
        "tail_risk_winner": winner,
        "cvar_minus_deterministic_mwh": float(cvar_diff),
        "paired_outcomes": {
            "probability_cvar_better": float(np.mean(differences < -tol)),
            "probability_deterministic_better": float(np.mean(differences > tol)),
            "probability_tie": float(np.mean(np.abs(differences) <= tol)),
            "mean_difference_mwh": float(np.mean(differences)),
            "minimum_difference_mwh": float(np.min(differences)),
            "maximum_difference_mwh": float(np.max(differences)),
        },
    }
    (results_dir / "annual_mc_risk_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    plot_annual_mc_results(summary, annual_df, hourly_df, visuals_dir)
    return summary
