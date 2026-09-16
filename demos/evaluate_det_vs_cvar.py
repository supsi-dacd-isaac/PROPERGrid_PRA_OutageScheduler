#!/usr/bin/env python3
"""Deterministic N-1 versus risk-informed CVaR outage scheduling.

This is a standalone experiment driver for the PROPER repository.  It uses the
existing ``utils.data_preporcess.data_loader`` only to load a pandapower network,
the hourly nodal demand table, and its JSON configuration.  Model construction,
scenario generation, schedule optimization, fixed-schedule out-of-sample
evaluation, KPIs, plots, and exports are implemented here.

The script deliberately supports three computational profiles:

* ``smoke``: 4 weeks at daily resolution, two seven-day line outages;
* ``pilot``: 13 weeks at weekly resolution, four planned line outages;
* ``full``: 52 weeks at weekly resolution and all eight notebook outages.

Examples (run from the repository root)::

    python demos/evaluate_det_vs_cvar.py  --system IEEE24 --profile smoke

    python demos/evaluate_det_vs_cvar.py --system IEEE24 --profile pilot --include-hybrid --run-scalability

    python demos/evaluate_det_vs_cvar.py  --system IEEE118 --profile pilot --train-scenarios 10 --validation-scenarios 100 --test-scenarios 500 --contingencies 10 --time-limit 3600

Requirements: numpy, pandas, matplotlib, pandapower, gurobipy and a working
Gurobi licence.  Input pickle files must be trusted because the repository data
loader may deserialize them.

Important interpretation
------------------------
The risk-equivalent scheduler replaces blanket N-1 enforcement with a CVaR
budget calibrated from the deterministic schedule on validation scenarios.  It
is the formulation that can test "lower cost at N-1-equivalent risk".  The
optional hybrid scheduler retains the deterministic N-1 constraints and adds
CVaR; it tests additional robustness and should not be expected to be cheaper.

When no contingency probability file is supplied, selected contingencies are
assigned equal conditional weights.  Such results are a controlled stress test,
not a real-world expected-risk estimate.  Supply failure probabilities for a
probabilistic claim.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("det_vs_cvar")


@dataclass(frozen=True)
class OutageTask:
    name: str
    kind: str  # "branch" or "generator"
    element: int
    duration_days: int
    cost_per_day: float
    priority: float


@dataclass(frozen=True)
class Contingency:
    name: str
    kind: str  # "branch" or "generator"
    element: int
    probability: float = 0.0


@dataclass
class NetworkData:
    n_bus: int
    n_branch: int
    n_gen: int
    base_mva: float
    from_bus: np.ndarray
    to_bus: np.ndarray
    branch_b: np.ndarray
    branch_shift_rad: np.ndarray
    branch_limit_mw: np.ndarray
    branch_status: np.ndarray
    gen_bus: np.ndarray
    p_min_mw: np.ndarray
    p_max_mw: np.ndarray
    gen_status: np.ndarray
    cost_quadratic: np.ndarray
    cost_linear: np.ndarray
    reference_bus: int
    cost_source: str


@dataclass(frozen=True)
class ExperimentSettings:
    profile: str
    resolution: str
    horizon_steps: int
    planned_outage_preset: str
    max_concurrent_outages: int
    n_contingencies: int
    n_train: int
    n_validation: int
    n_test: int
    alpha: float
    risk_budget_factors: tuple[float, ...]
    time_limit_s: float
    mip_gap: float
    evaluation_batch_size: int
    n_n1_envelope_scenarios: int


PROFILES: dict[str, ExperimentSettings] = {
    "smoke": ExperimentSettings(
        profile="smoke",
        resolution="D",
        horizon_steps=28,
        planned_outage_preset="smoke",
        max_concurrent_outages=1,
        n_contingencies=5,
        n_train=10,
        n_validation=50,
        n_test=200,
        alpha=0.90,
        risk_budget_factors=(0.8, 1.0, 1.2),
        time_limit_s=300.0,
        mip_gap=0.02,
        evaluation_batch_size=10,
        n_n1_envelope_scenarios=10,
    ),
    "pilot": ExperimentSettings(
        profile="pilot",
        resolution="W",
        horizon_steps=13,
        planned_outage_preset="pilot",
        max_concurrent_outages=1,
        n_contingencies=10,
        n_train=25,
        n_validation=200,
        n_test=500,
        alpha=0.90,
        risk_budget_factors=(0.8, 1.0, 1.2),
        time_limit_s=900.0,
        mip_gap=0.02,
        evaluation_batch_size=10,
        n_n1_envelope_scenarios=20,
    ),
    "full": ExperimentSettings(
        profile="full",
        resolution="W",
        horizon_steps=52,
        planned_outage_preset="full",
        max_concurrent_outages=2,
        n_contingencies=30,
        n_train=100,
        n_validation=500,
        n_test=2000,
        alpha=0.95,
        risk_budget_factors=(0.8, 0.9, 1.0, 1.1),
        time_limit_s=3600.0,
        mip_gap=0.01,
        evaluation_batch_size=10,
        n_n1_envelope_scenarios=20,
    ),
}


DEFAULT_TASKS = (
    OutageTask("line_0", "branch", 0, 25, 1000.0, 1.0),
    OutageTask("line_1", "branch", 1, 7, 1000.0, 2.0),
    OutageTask("line_2", "branch", 2, 55, 1000.0, 1.0),
    OutageTask("line_3", "branch", 3, 7, 1000.0, 3.0),
    OutageTask("line_5", "branch", 5, 30, 1000.0, 2.0),
    OutageTask("line_8", "branch", 8, 30, 1000.0, 1.0),
    OutageTask("gen_1", "generator", 1, 30, 5000.0, 1.0),
    OutageTask("gen_2", "generator", 2, 60, 5000.0, 3.0),
)


def require_gurobi() -> tuple[Any, Any]:
    try:
        gp = importlib.import_module("gurobipy")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gurobipy is required. Install it in the project environment and "
            "verify that a Gurobi licence is available."
        ) from exc
    return gp, gp.GRB


def repository_root() -> Path:
    script = Path(__file__).resolve()
    candidates = (script.parent.parent, Path.cwd(), Path.cwd().parent)
    for candidate in candidates:
        if (candidate / "config").is_dir() or (candidate / "data").is_dir():
            return candidate
    return script.parent.parent


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required")
    return values


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Use a comma-separated list of positive integers")
    return values


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_repository_case(config_path: Path) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Use the repository loader while keeping this evaluation self-contained."""
    root = repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("utils.data_preporcess")
        loader = getattr(module, "data_loader")
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "Could not import utils.data_preporcess.data_loader. Place this "
            "script inside the PROPER repository (for example in "
            "probabilistic_model/) and run it from the repository root."
        ) from exc

    network, hourly_demand, config = loader(str(config_path.resolve()))
    frame = orient_demand(hourly_demand, len(network.bus))
    return network, frame, dict(config)


def orient_demand(raw: Any, expected_buses: int) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    elif isinstance(raw, pd.Series):
        frame = raw.to_frame()
    else:
        array = np.asarray(raw)
        if array.ndim != 2:
            raise ValueError(f"Demand must be two-dimensional; obtained {array.shape}")
        frame = pd.DataFrame(array)

    rows, columns = frame.shape
    if rows == expected_buses and columns != expected_buses:
        frame = frame.T
    elif columns != expected_buses and rows < columns:
        frame = frame.T

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(axis=0, limit_direction="both").ffill().bfill()
    if frame.shape[1] != expected_buses:
        raise ValueError(
            f"Demand has {frame.shape[1]} nodal columns after orientation; "
            f"the network has {expected_buses} buses."
        )
    if frame.isna().any().any():
        raise ValueError("Demand contains unresolved missing values")
    return frame.astype(float)


def ensure_power_case(network: Any) -> dict[str, Any]:
    ppc = getattr(network, "_ppc", None)
    if not isinstance(ppc, dict) or not all(key in ppc for key in ("bus", "branch", "gen")):
        try:
            pp = importlib.import_module("pandapower")
        except ModuleNotFoundError as exc:
            raise RuntimeError("pandapower is required to initialize the network case") from exc
        pp.runpp(network, calculate_voltage_angles=True, init="flat")
        ppc = network._ppc
    return ppc


def fallback_branch_limits(network: Any, n_branch: int, default_limit: float) -> np.ndarray:
    limits: list[float] = []
    if hasattr(network, "line"):
        for _, row in network.line.iterrows():
            max_i = float(row.get("max_i_ka", np.nan))
            limits.append(175.0 if np.isfinite(max_i) and max_i <= 1.0 else 500.0)
    if hasattr(network, "trafo"):
        limits.extend([400.0] * len(network.trafo))
    if len(limits) < n_branch:
        limits.extend([default_limit] * (n_branch - len(limits)))
    return np.asarray(limits[:n_branch], dtype=float)


def extract_generation_costs(ppc: dict[str, Any], n_gen: int) -> tuple[np.ndarray, np.ndarray, str]:
    quadratic = np.zeros(n_gen, dtype=float)
    linear = np.ones(n_gen, dtype=float)
    table = ppc.get("gencost")
    if table is None or len(table) < n_gen:
        return quadratic, linear, "fallback unit generation cost"

    used_fallback = False
    for generator in range(n_gen):
        row = np.asarray(table[generator], dtype=float)
        model_type = int(row[0])
        n_cost = int(row[3])
        if model_type == 2 and n_cost >= 1:
            coefficients = row[4 : 4 + n_cost]
            degree = n_cost - 1
            if degree <= 2:
                if degree == 2:
                    quadratic[generator] = coefficients[0]
                    linear[generator] = coefficients[1]
                elif degree == 1:
                    linear[generator] = coefficients[0]
                else:
                    linear[generator] = 0.0
            else:
                # A linear marginal-cost approximation avoids a non-quadratic MIP.
                used_fallback = True
                polynomial = np.poly1d(coefficients)
                derivative = np.polyder(polynomial)
                linear[generator] = max(float(derivative(0.0)), 0.0)
        elif model_type == 1 and n_cost >= 2:
            points = row[4 : 4 + 2 * n_cost].reshape(-1, 2)
            delta_p = points[-1, 0] - points[0, 0]
            linear[generator] = (
                (points[-1, 1] - points[0, 1]) / delta_p if abs(delta_p) > 1e-12 else 1.0
            )
            used_fallback = True
        else:
            used_fallback = True
    source = "MATPOWER/pandapower gencost"
    if used_fallback:
        source += " with linear approximations for unsupported rows"
    return quadratic, linear, source


def extract_network_data(network: Any, default_branch_limit: float) -> NetworkData:
    ppc = ensure_power_case(network)
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)
    base_mva = float(ppc.get("baseMVA", 100.0))

    bus_ids = bus[:, 0].astype(int)
    bus_position = {identifier: position for position, identifier in enumerate(bus_ids)}
    from_bus = np.array([bus_position[int(value)] for value in branch[:, 0]], dtype=int)
    to_bus = np.array([bus_position[int(value)] for value in branch[:, 1]], dtype=int)
    reactance = branch[:, 3].copy()
    reactance[np.abs(reactance) < 1e-8] = np.sign(reactance[np.abs(reactance) < 1e-8]) * 1e-8
    reactance[reactance == 0] = 1e-8
    tap = branch[:, 8].copy()
    tap[np.abs(tap) < 1e-12] = 1.0
    susceptance = base_mva / (reactance * tap)
    shift = np.deg2rad(branch[:, 9])

    fallback = fallback_branch_limits(network, len(branch), default_branch_limit)
    rate = branch[:, 5].copy()
    invalid_rate = ~np.isfinite(rate) | (rate <= 0)
    if invalid_rate.any():
        LOGGER.warning(
            "%d/%d PPC branches have no positive RATE_A; using the notebook-compatible fallback limits",
            int(invalid_rate.sum()),
            len(rate),
        )
    rate[invalid_rate] = fallback[invalid_rate]

    gen_bus = np.array([bus_position[int(value)] for value in gen[:, 0]], dtype=int)
    gen_status = gen[:, 7] > 0.5
    p_max = np.maximum(gen[:, 8], 0.0) * gen_status
    p_min = np.minimum(np.maximum(gen[:, 9], 0.0), p_max) * gen_status
    quadratic, linear, cost_source = extract_generation_costs(ppc, len(gen))

    reference_candidates = np.flatnonzero(bus[:, 1].astype(int) == 3)
    reference_bus = int(reference_candidates[0]) if len(reference_candidates) else 0
    return NetworkData(
        n_bus=len(bus),
        n_branch=len(branch),
        n_gen=len(gen),
        base_mva=base_mva,
        from_bus=from_bus,
        to_bus=to_bus,
        branch_b=susceptance,
        branch_shift_rad=shift,
        branch_limit_mw=rate,
        branch_status=branch[:, 10] > 0.5,
        gen_bus=gen_bus,
        p_min_mw=p_min,
        p_max_mw=p_max,
        gen_status=gen_status,
        cost_quadratic=quadratic,
        cost_linear=linear,
        reference_bus=reference_bus,
        cost_source=cost_source,
    )


def map_demand_to_internal_buses(network: Any, demand: pd.DataFrame, n_internal: int) -> np.ndarray:
    values = demand.to_numpy(dtype=float)
    if values.shape[1] == len(network.bus):
        lookup = getattr(network, "_pd2ppc_lookups", {}).get("bus")
        if lookup is None:
            if values.shape[1] == n_internal:
                return values
            raise ValueError("The pandapower bus lookup is unavailable for fused buses")
        mapped = np.zeros((len(values), n_internal), dtype=float)
        for source_position, bus_index in enumerate(network.bus.index):
            try:
                target = int(lookup[int(bus_index)])
            except (IndexError, KeyError, TypeError):
                if values.shape[1] == n_internal:
                    return values
                raise ValueError(f"Cannot map pandapower bus index {bus_index}")
            if 0 <= target < n_internal:
                mapped[:, target] += values[:, source_position]
        return mapped
    if values.shape[1] == n_internal:
        return values
    raise ValueError(
        f"Cannot map {values.shape[1]} demand columns to {n_internal} internal buses"
    )


def aggregate_demand(hourly: np.ndarray, resolution: str, representative: str) -> tuple[np.ndarray, float]:
    step_hours = {"H": 1, "D": 24, "W": 168}[resolution]
    usable = len(hourly) - len(hourly) % step_hours
    if usable < step_hours:
        raise ValueError(f"Not enough hourly data for resolution {resolution}")
    blocks = hourly[:usable].reshape(-1, step_hours, hourly.shape[1])
    if representative == "mean":
        aggregated = blocks.mean(axis=1)
    elif representative == "system_peak":
        indices = blocks.sum(axis=2).argmax(axis=1)
        aggregated = blocks[np.arange(len(blocks)), indices]
    else:
        raise ValueError(f"Unknown representative method {representative}")
    return np.maximum(aggregated, 0.0), float(step_hours)


def select_window(data: np.ndarray, horizon: int, method: str) -> tuple[np.ndarray, int]:
    if horizon > len(data):
        raise ValueError(f"Requested {horizon} steps but only {len(data)} are available")
    if method == "first":
        start = 0
    elif method == "latest":
        start = len(data) - horizon
    elif method == "peak":
        totals = data.sum(axis=1)
        cumulative = np.r_[0.0, np.cumsum(totals)]
        scores = cumulative[horizon:] - cumulative[:-horizon]
        start = int(np.argmax(scores))
    else:
        raise ValueError(f"Unknown window method {method}")
    return data[start : start + horizon].copy(), start


def residual_pool(history: np.ndarray, resolution: str) -> np.ndarray:
    preferred_lag = {"H": 168, "D": 7, "W": 4}[resolution]
    lag = min(preferred_lag, max(1, len(history) // 4))
    if len(history) > lag:
        residual = history[lag:] - history[:-lag]
    else:
        residual = history - history.mean(axis=0, keepdims=True)
    residual = residual - residual.mean(axis=0, keepdims=True)
    return residual


def generate_load_scenarios(
    base: np.ndarray,
    history: np.ndarray,
    n_scenarios: int,
    family: str,
    seed: int,
    resolution: str,
    student_df: float,
    block_length: int,
    residual_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    residuals = residual_pool(history, resolution)
    scenarios = np.empty((n_scenarios, *base.shape), dtype=float)
    block_length = max(1, min(block_length, len(base), len(residuals)))

    if family == "empirical":
        for scenario in range(n_scenarios):
            sampled: list[np.ndarray] = []
            while sum(len(block) for block in sampled) < len(base):
                start = int(rng.integers(0, max(1, len(residuals) - block_length + 1)))
                sampled.append(residuals[start : start + block_length])
            innovation = np.concatenate(sampled, axis=0)[: len(base)]
            scenarios[scenario] = np.maximum(base + residual_scale * innovation, 0.0)
    elif family == "student_t":
        covariance = np.cov(residuals, rowvar=False)
        if np.ndim(covariance) == 0:
            covariance = np.array([[float(covariance)]])
        diagonal = np.diag(np.diag(covariance))
        covariance = 0.8 * covariance + 0.2 * diagonal
        jitter = max(float(np.trace(covariance)) / max(base.shape[1], 1), 1.0) * 1e-8
        covariance = covariance + jitter * np.eye(base.shape[1])
        total = residuals.sum(axis=1)
        rho = 0.0
        if len(total) > 2 and np.std(total[:-1]) > 1e-12 and np.std(total[1:]) > 1e-12:
            rho = float(np.clip(np.corrcoef(total[:-1], total[1:])[0, 1], -0.8, 0.8))
        for scenario in range(n_scenarios):
            gaussian = rng.multivariate_normal(np.zeros(base.shape[1]), covariance, size=len(base))
            scale = np.sqrt(rng.chisquare(student_df, size=len(base)) / student_df)
            innovation = gaussian / scale[:, None]
            for step in range(1, len(base)):
                innovation[step] = rho * innovation[step - 1] + math.sqrt(1 - rho**2) * innovation[step]
            scenarios[scenario] = np.maximum(base + residual_scale * innovation, 0.0)
    else:
        raise ValueError(f"Unknown scenario family: {family}")
    return scenarios


def load_scenario_array(path: Optional[Path], expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        if "scenarios" in archive:
            array = archive["scenarios"]
        else:
            arrays = [archive[key] for key in archive.files if archive[key].ndim == 3]
            if not arrays:
                raise ValueError(f"No 3-D scenario array in {path}")
            array = arrays[0]
    else:
        raise ValueError("Scenario files must be .npy or .npz")
    array = np.asarray(array, dtype=float)
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_shape:
        raise ValueError(
            f"Scenario array {path} has shape {array.shape}; expected (S, {expected_shape[0]}, {expected_shape[1]})"
        )
    return np.maximum(array, 0.0)


def empirical_var_cvar(losses: Sequence[float], alpha: float) -> tuple[float, float]:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("CVaR requires a finite, non-empty one-dimensional sample")
    value_at_risk = float(np.quantile(values, alpha, method="higher"))
    cvar = value_at_risk + float(np.maximum(values - value_at_risk, 0.0).mean()) / (1.0 - alpha)
    return value_at_risk, cvar


def paired_bootstrap_interval(
    first: Sequence[float],
    second: Sequence[float],
    statistic: str,
    alpha: float,
    seed: int,
    repetitions: int,
) -> tuple[float, float, float]:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired samples must have the same shape")
    rng = np.random.default_rng(seed)

    def calculate(x: np.ndarray, y: np.ndarray) -> float:
        if statistic == "mean":
            return float(np.mean(y) - np.mean(x))
        if statistic == "cvar":
            return empirical_var_cvar(y, alpha)[1] - empirical_var_cvar(x, alpha)[1]
        raise ValueError(statistic)

    estimate = calculate(a, b)
    boot = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.integers(0, len(a), size=len(a))
        boot[index] = calculate(a[sample], b[sample])
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def choose_tasks(preset: str, n_branch: int, n_gen: int) -> list[OutageTask]:
    if preset == "smoke":
        wanted = {"line_1", "line_3"}
    elif preset == "pilot":
        wanted = {"line_0", "line_1", "line_3", "line_5"}
    elif preset == "full":
        wanted = {task.name for task in DEFAULT_TASKS}
    else:
        raise ValueError(f"Unknown outage preset: {preset}")
    selected = [task for task in DEFAULT_TASKS if task.name in wanted]
    for task in selected:
        limit = n_branch if task.kind == "branch" else n_gen
        if not 0 <= task.element < limit:
            raise IndexError(f"Task {task.name} targets unavailable {task.kind} {task.element}")
    return selected


def load_tasks(path: Optional[Path], preset: str, n_branch: int, n_gen: int) -> list[OutageTask]:
    if path is None:
        return choose_tasks(preset, n_branch, n_gen)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("outages", payload) if isinstance(payload, dict) else payload
    tasks = [OutageTask(**row) for row in rows]
    for task in tasks:
        if task.kind not in {"branch", "generator"}:
            raise ValueError(f"Invalid task kind in {task}")
        limit = n_branch if task.kind == "branch" else n_gen
        if not 0 <= task.element < limit:
            raise IndexError(f"Task {task.name} targets unavailable element")
    return tasks


def remap_default_generator_tasks(
    tasks: Sequence[OutageTask],
    network_object: Any,
    custom_plan_supplied: bool,
    n_internal_generators: int,
) -> list[OutageTask]:
    """Map notebook ``network.gen`` ids to pandapower/PPC generator rows.

    A custom outage JSON is defined directly in internal PPC index space and is
    therefore left unchanged.  The built-in tasks reproduce the notebook,
    where ``gen_1`` and ``gen_2`` refer to ``network.gen`` indices.
    """
    if custom_plan_supplied or not hasattr(network_object, "gen"):
        return list(tasks)
    external_grid_count = len(getattr(network_object, "ext_grid", []))
    generator_indices = list(network_object.gen.index)
    mapped: list[OutageTask] = []
    for task in tasks:
        if task.kind != "generator":
            mapped.append(task)
            continue
        if task.element not in generator_indices:
            raise IndexError(
                f"Default task {task.name} refers to network.gen index {task.element}, "
                f"which is not present: {generator_indices}"
            )
        internal = external_grid_count + generator_indices.index(task.element)
        if not 0 <= internal < n_internal_generators:
            raise IndexError(
                f"Mapped task {task.name} to PPC generator row {internal}, outside "
                f"the {n_internal_generators} available rows"
            )
        mapped.append(
            OutageTask(
                task.name,
                task.kind,
                internal,
                task.duration_days,
                task.cost_per_day,
                task.priority,
            )
        )
    return mapped


def task_duration_steps(task: OutageTask, step_hours: float) -> int:
    return max(1, int(math.ceil(24.0 * task.duration_days / step_hours)))


def candidate_contingencies(network: NetworkData, include_generators: bool) -> list[Contingency]:
    contingencies = [
        Contingency(f"branch_{index}", "branch", index)
        for index in range(network.n_branch)
        if network.branch_status[index]
    ]
    if include_generators:
        contingencies.extend(
            Contingency(f"generator_{index}", "generator", index)
            for index in range(network.n_gen)
            if network.gen_status[index] and network.p_max_mw[index] > 0
        )
    return contingencies


def load_contingency_probabilities(path: Optional[Path]) -> dict[str, float]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    required = {"name", "probability"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    probabilities = dict(zip(table["name"].astype(str), table["probability"].astype(float)))
    if any(value < 0 for value in probabilities.values()):
        raise ValueError("Contingency probabilities cannot be negative")
    return probabilities


def assign_probabilities(
    contingencies: Sequence[Contingency],
    supplied: dict[str, float],
    no_contingency_probability: float,
) -> tuple[list[Contingency], float, str]:
    if not 0.0 <= no_contingency_probability < 1.0:
        raise ValueError("no-contingency probability must be in [0, 1)")
    if supplied:
        base_probability = float(supplied.get("base", supplied.get("no_contingency", 0.0)))
        raw = np.array([supplied.get(item.name, 0.0) for item in contingencies], dtype=float)
        total = base_probability + raw.sum()
        if total <= 0:
            raise ValueError("The supplied contingency probabilities sum to zero")
        base_probability /= total
        raw /= total
        interpretation = "user-supplied normalized contingency probabilities"
    else:
        base_probability = no_contingency_probability
        raw = np.full(len(contingencies), (1.0 - base_probability) / max(len(contingencies), 1))
        interpretation = "uniform conditional contingency weights (stress-test assumption)"
    weighted = [
        Contingency(item.name, item.kind, item.element, float(probability))
        for item, probability in zip(contingencies, raw)
    ]
    return weighted, base_probability, interpretation


def sample_joint_contingencies(
    contingencies: Sequence[Contingency],
    base_probability: float,
    n_scenarios: int,
    seed: int,
) -> list[Optional[Contingency]]:
    rng = np.random.default_rng(seed)
    options: list[Optional[Contingency]] = [None, *contingencies]
    probabilities = np.array([base_probability, *[item.probability for item in contingencies]], dtype=float)
    probabilities /= probabilities.sum()
    indices = rng.choice(len(options), size=n_scenarios, replace=True, p=probabilities)
    return [options[int(index)] for index in indices]


@dataclass
class StateHandle:
    label: str
    generation_cost: Any
    ens_mwh: Any
    spilled_mwh: Any
    flow: Any


@dataclass
class ScheduleSolution:
    name: str
    schedule: pd.DataFrame
    starts: pd.DataFrame
    maintenance_cost: float
    priority_delay_score: float
    training_cvar_mwh: float
    status: str
    objective: float
    best_bound: float
    mip_gap: float
    build_time_s: float
    solve_time_s: float
    num_variables: int
    num_constraints: int
    num_nonzeros: int


@dataclass(frozen=True)
class EvaluationRequest:
    label: str
    scenario: int
    variant: str
    demand: np.ndarray
    contingency: Optional[Contingency]


def validate_tasks(
    tasks: Sequence[OutageTask],
    horizon_steps: int,
    step_hours: float,
    max_concurrent: int,
) -> None:
    seen: set[tuple[str, int]] = set()
    for task in tasks:
        target = (task.kind, task.element)
        if target in seen:
            raise ValueError(f"Multiple planned tasks target {target}; this is unsupported")
        seen.add(target)
        duration = task_duration_steps(task, step_hours)
        if duration > horizon_steps:
            raise ValueError(
                f"Task {task.name} requires {duration} steps but the horizon has {horizon_steps}. "
                "Use a longer horizon or a smaller outage plan."
            )
    required_capacity = sum(task_duration_steps(task, step_hours) for task in tasks)
    available_capacity = horizon_steps * max_concurrent
    if required_capacity > available_capacity:
        raise ValueError(
            f"The tasks require {required_capacity} outage-steps but only "
            f"{available_capacity} are available with max_concurrent={max_concurrent}."
        )


def add_schedule_variables(
    model: Any,
    tasks: Sequence[OutageTask],
    horizon_steps: int,
    step_hours: float,
    max_concurrent: int,
    gp: Any,
    GRB: Any,
) -> tuple[Any, Any, Any]:
    task_names = [task.name for task in tasks]
    active = model.addVars(range(horizon_steps), task_names, vtype=GRB.BINARY, name="outage_active")
    start_keys: list[tuple[int, str]] = []
    duration = {task.name: task_duration_steps(task, step_hours) for task in tasks}
    for task in tasks:
        start_keys.extend((step, task.name) for step in range(horizon_steps - duration[task.name] + 1))
    starts = model.addVars(start_keys, vtype=GRB.BINARY, name="outage_start")

    for task in tasks:
        valid_starts = range(horizon_steps - duration[task.name] + 1)
        model.addConstr(
            gp.quicksum(starts[step, task.name] for step in valid_starts) == 1,
            name=f"schedule_once_{sanitize(task.name)}",
        )
        for time_index in range(horizon_steps):
            covering = [
                step
                for step in valid_starts
                if step <= time_index < step + duration[task.name]
            ]
            model.addConstr(
                active[time_index, task.name]
                == gp.quicksum(starts[step, task.name] for step in covering),
                name=f"active_link_{sanitize(task.name)}_{time_index}",
            )

    model.addConstrs(
        (
            gp.quicksum(active[time_index, task.name] for task in tasks) <= max_concurrent
            for time_index in range(horizon_steps)
        ),
        name="max_concurrent_outages",
    )

    priority_delay = gp.quicksum(
        task.priority
        * (step / max(horizon_steps - 1, 1))
        * starts[step, task.name]
        for task in tasks
        for step in range(horizon_steps - duration[task.name] + 1)
    )
    return active, starts, priority_delay


def planned_indicator(
    time_index: int,
    kind: str,
    element: int,
    tasks: Sequence[OutageTask],
    active: Any,
) -> Any:
    matching = [task.name for task in tasks if task.kind == kind and task.element == element]
    if not matching:
        return 0.0
    if isinstance(active, pd.DataFrame):
        return float(active.loc[matching[0], f"step_{time_index}"])
    return active[time_index, matching[0]]


def add_operating_state(
    model: Any,
    label: str,
    network: NetworkData,
    demand: np.ndarray,
    tasks: Sequence[OutageTask],
    active: Any,
    contingency: Optional[Contingency],
    step_hours: float,
    angle_bound_rad: float,
    gp: Any,
    GRB: Any,
) -> StateHandle:
    horizon_steps, n_bus = demand.shape
    if n_bus != network.n_bus:
        raise ValueError(f"State {label} has {n_bus} buses; expected {network.n_bus}")
    safe = sanitize(label)
    pgen = model.addVars(range(horizon_steps), range(network.n_gen), lb=0.0, name=f"p_{safe}")
    theta = model.addVars(
        range(horizon_steps),
        range(network.n_bus),
        lb=-angle_bound_rad,
        ub=angle_bound_rad,
        name=f"theta_{safe}",
    )
    flow = model.addVars(
        range(horizon_steps),
        range(network.n_branch),
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name=f"flow_{safe}",
    )
    shed = model.addVars(range(horizon_steps), range(network.n_bus), lb=0.0, name=f"shed_{safe}")
    spill = model.addVars(range(horizon_steps), range(network.n_bus), lb=0.0, name=f"spill_{safe}")

    generators_at_bus = {
        bus: np.flatnonzero(network.gen_bus == bus).tolist() for bus in range(network.n_bus)
    }
    outgoing = {
        bus: np.flatnonzero(network.from_bus == bus).tolist() for bus in range(network.n_bus)
    }
    incoming = {
        bus: np.flatnonzero(network.to_bus == bus).tolist() for bus in range(network.n_bus)
    }

    generation_cost = gp.QuadExpr()
    ens_mwh = gp.LinExpr()
    spilled_mwh = gp.LinExpr()
    for time_index in range(horizon_steps):
        model.addConstr(theta[time_index, network.reference_bus] == 0.0, name=f"ref_{safe}_{time_index}")

        for generator in range(network.n_gen):
            planned = planned_indicator(time_index, "generator", generator, tasks, active)
            failed = (
                not bool(network.gen_status[generator])
                or (
                    contingency is not None
                    and contingency.kind == "generator"
                    and contingency.element == generator
                )
            )
            if failed:
                model.addConstr(pgen[time_index, generator] == 0.0, name=f"gen_off_{safe}_{time_index}_{generator}")
            else:
                model.addConstr(
                    pgen[time_index, generator] <= network.p_max_mw[generator] * (1.0 - planned),
                    name=f"gen_up_{safe}_{time_index}_{generator}",
                )
                model.addConstr(
                    pgen[time_index, generator] >= network.p_min_mw[generator] * (1.0 - planned),
                    name=f"gen_low_{safe}_{time_index}_{generator}",
                )
            generation_cost += step_hours * (
                network.cost_quadratic[generator] * pgen[time_index, generator] * pgen[time_index, generator]
                + network.cost_linear[generator] * pgen[time_index, generator]
            )

        for branch in range(network.n_branch):
            planned = planned_indicator(time_index, "branch", branch, tasks, active)
            failed = (
                not bool(network.branch_status[branch])
                or (
                    contingency is not None
                    and contingency.kind == "branch"
                    and contingency.element == branch
                )
            )
            if failed:
                model.addConstr(flow[time_index, branch] == 0.0, name=f"branch_off_{safe}_{time_index}_{branch}")
                relaxation = 1.0
            else:
                model.addConstr(
                    flow[time_index, branch] <= network.branch_limit_mw[branch] * (1.0 - planned),
                    name=f"branch_up_{safe}_{time_index}_{branch}",
                )
                model.addConstr(
                    flow[time_index, branch] >= -network.branch_limit_mw[branch] * (1.0 - planned),
                    name=f"branch_low_{safe}_{time_index}_{branch}",
                )
                relaxation = planned

            dc_flow = network.branch_b[branch] * (
                theta[time_index, network.from_bus[branch]]
                - theta[time_index, network.to_bus[branch]]
                - network.branch_shift_rad[branch]
            )
            big_m = (
                abs(network.branch_b[branch])
                * (2.0 * angle_bound_rad + abs(network.branch_shift_rad[branch]))
                + network.branch_limit_mw[branch]
            )
            model.addConstr(
                flow[time_index, branch] - dc_flow <= big_m * relaxation,
                name=f"dc_up_{safe}_{time_index}_{branch}",
            )
            model.addConstr(
                flow[time_index, branch] - dc_flow >= -big_m * relaxation,
                name=f"dc_low_{safe}_{time_index}_{branch}",
            )

        for bus in range(network.n_bus):
            generation = gp.quicksum(pgen[time_index, gen] for gen in generators_at_bus[bus])
            net_export = (
                gp.quicksum(flow[time_index, branch] for branch in outgoing[bus])
                - gp.quicksum(flow[time_index, branch] for branch in incoming[bus])
            )
            model.addConstr(
                generation
                + shed[time_index, bus]
                - spill[time_index, bus]
                - float(demand[time_index, bus])
                == net_export,
                name=f"balance_{safe}_{time_index}_{bus}",
            )
            model.addConstr(
                shed[time_index, bus] <= float(demand[time_index, bus]),
                name=f"shed_cap_{safe}_{time_index}_{bus}",
            )
            ens_mwh += step_hours * shed[time_index, bus]
            spilled_mwh += step_hours * spill[time_index, bus]

    return StateHandle(label, generation_cost, ens_mwh, spilled_mwh, flow)


def apply_solver_parameters(
    model: Any,
    time_limit_s: float,
    mip_gap: float,
    threads: int,
    solver_log: bool,
    is_mip: bool,
) -> None:
    model.Params.OutputFlag = 1 if solver_log else 0
    model.Params.TimeLimit = float(time_limit_s)
    model.Params.Threads = int(max(1, threads))
    model.Params.Presolve = 2
    if is_mip:
        model.Params.MIPGap = float(mip_gap)
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.5
        model.Params.NodefileStart = 0.5


def status_name(status: int, GRB: Any) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, f"STATUS_{status}")


def extract_schedule(
    active: Any,
    starts: Any,
    tasks: Sequence[OutageTask],
    horizon_steps: int,
    step_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.DataFrame(
        {
            f"step_{time_index}": [float(active[time_index, task.name].X) for task in tasks]
            for time_index in range(horizon_steps)
        },
        index=[task.name for task in tasks],
    )
    schedule.index.name = "outage"
    start_records = []
    for task in tasks:
        duration = task_duration_steps(task, step_hours)
        selected = [
            step
            for step in range(horizon_steps - duration + 1)
            if starts[step, task.name].X > 0.5
        ]
        start = selected[0] if selected else int(np.argmax(schedule.loc[task.name].to_numpy()))
        start_records.append(
            {
                "outage": task.name,
                "kind": task.kind,
                "element": task.element,
                "start_step": start,
                "duration_steps": duration,
                "start_hour": start * step_hours,
                "duration_days": task.duration_days,
                "priority": task.priority,
            }
        )
    return schedule, pd.DataFrame(start_records)


def add_cvar(
    model: Any,
    losses: Sequence[Any],
    alpha: float,
    gp: Any,
    GRB: Any,
) -> tuple[Any, Any, Any]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    eta = model.addVar(lb=0.0, name="cvar_eta")
    excess = model.addVars(range(len(losses)), lb=0.0, name="cvar_excess")
    model.addConstrs(
        (excess[index] >= loss - eta for index, loss in enumerate(losses)),
        name="cvar_excess_definition",
    )
    cvar = eta + gp.quicksum(excess[index] for index in range(len(losses))) / (
        (1.0 - alpha) * len(losses)
    )
    return eta, excess, cvar


def solve_schedule_model(
    name: str,
    mode: str,
    network: NetworkData,
    base_demand: np.ndarray,
    tasks: Sequence[OutageTask],
    hard_contingencies: Sequence[Contingency],
    risk_demands: Optional[np.ndarray],
    risk_contingencies: Optional[Sequence[Optional[Contingency]]],
    step_hours: float,
    max_concurrent: int,
    alpha: float,
    risk_budget_mwh: Optional[float],
    risk_weight: float,
    priority_weight: float,
    voll: float,
    spill_penalty: float,
    strict_n1: bool,
    shed_tolerance_mwh: float,
    must_protect_count: int,
    time_limit_s: float,
    mip_gap: float,
    threads: int,
    solver_log: bool,
    angle_bound_rad: float,
) -> ScheduleSolution:
    gp, GRB = require_gurobi()
    build_started = time.perf_counter()
    model = gp.Model(sanitize(name))
    active, starts, priority_delay = add_schedule_variables(
        model, tasks, len(base_demand), step_hours, max_concurrent, gp, GRB
    )
    base = add_operating_state(
        model,
        "expected_base",
        network,
        base_demand,
        tasks,
        active,
        None,
        step_hours,
        angle_bound_rad,
        gp,
        GRB,
    )
    # All compared formulations must serve the expected normal operating state.
    model.addConstr(base.ens_mwh <= shed_tolerance_mwh, name="serve_expected_base")

    hard_states: list[StateHandle] = []
    protected = list(hard_contingencies)
    if mode == "risk_equivalent":
        protected = protected[:must_protect_count]
    for contingency in protected:
        state = add_operating_state(
            model,
            f"hard_{contingency.name}",
            network,
            base_demand,
            tasks,
            active,
            contingency,
            step_hours,
            angle_bound_rad,
            gp,
            GRB,
        )
        hard_states.append(state)
        if strict_n1 or (mode == "risk_equivalent" and must_protect_count > 0):
            model.addConstr(
                state.ens_mwh <= shed_tolerance_mwh,
                name=f"strict_{sanitize(contingency.name)}",
            )

    risk_states: list[StateHandle] = []
    if mode in {"risk_equivalent", "hybrid"}:
        if risk_demands is None or risk_contingencies is None:
            raise ValueError(f"{mode} requires risk scenarios")
        if len(risk_demands) != len(risk_contingencies):
            raise ValueError("Demand and contingency scenario counts do not match")
        for scenario, (demand, sampled_contingency) in enumerate(
            zip(risk_demands, risk_contingencies)
        ):
            contingency = sampled_contingency if mode == "risk_equivalent" else None
            risk_states.append(
                add_operating_state(
                    model,
                    f"risk_{scenario}",
                    network,
                    demand,
                    tasks,
                    active,
                    contingency,
                    step_hours,
                    angle_bound_rad,
                    gp,
                    GRB,
                )
            )

    maintenance_cost = float(sum(task.cost_per_day * task.duration_days for task in tasks))
    objective = priority_weight * priority_delay + base.generation_cost
    hard_ens = gp.quicksum(state.ens_mwh for state in hard_states) / max(len(hard_states), 1)
    objective += voll * (base.ens_mwh + hard_ens)
    objective += spill_penalty * (
        base.spilled_mwh
        + gp.quicksum(state.spilled_mwh for state in hard_states) / max(len(hard_states), 1)
    )

    cvar_expression: Any = 0.0
    if risk_states:
        expected_generation = gp.quicksum(state.generation_cost for state in risk_states) / len(risk_states)
        expected_ens = gp.quicksum(state.ens_mwh for state in risk_states) / len(risk_states)
        expected_spill = gp.quicksum(state.spilled_mwh for state in risk_states) / len(risk_states)
        objective = priority_weight * priority_delay + expected_generation
        objective += voll * expected_ens + spill_penalty * expected_spill
        _, _, cvar_expression = add_cvar(
            model, [state.ens_mwh for state in risk_states], alpha, gp, GRB
        )
        if risk_budget_mwh is not None:
            model.addConstr(cvar_expression <= float(risk_budget_mwh), name="risk_budget")
        if risk_weight > 0:
            objective += risk_weight * cvar_expression
        if mode == "hybrid":
            # Preserve the expected-load N-1 requirements in addition to CVaR.
            objective += voll * (base.ens_mwh + hard_ens)

    model.setObjective(objective, GRB.MINIMIZE)
    apply_solver_parameters(model, time_limit_s, mip_gap, threads, solver_log, is_mip=True)
    if np.any(network.cost_quadratic < -1e-12):
        model.Params.NonConvex = 2
    model.update()
    build_time = time.perf_counter() - build_started
    num_variables = int(model.NumVars)
    num_constraints = int(model.NumConstrs)
    num_nonzeros = int(model.NumNZs)
    solve_started = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - solve_started
    status = status_name(model.Status, GRB)
    if model.SolCount < 1:
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write(f"{sanitize(name)}_infeasible.ilp")
        model.dispose()
        raise RuntimeError(f"{name} ended with {status} and no incumbent solution")

    schedule, start_table = extract_schedule(active, starts, tasks, len(base_demand), step_hours)
    objective_value = float(model.ObjVal)
    best_bound = float(model.ObjBound)
    gap = float(model.MIPGap) if model.IsMIP else 0.0
    delay_value = float(priority_delay.getValue())
    training_cvar = float(cvar_expression.getValue()) if risk_states else float("nan")
    model.dispose()
    return ScheduleSolution(
        name=name,
        schedule=schedule,
        starts=start_table,
        maintenance_cost=maintenance_cost,
        priority_delay_score=delay_value,
        training_cvar_mwh=training_cvar,
        status=status,
        objective=objective_value,
        best_bound=best_bound,
        mip_gap=gap,
        build_time_s=build_time,
        solve_time_s=solve_time,
        num_variables=num_variables,
        num_constraints=num_constraints,
        num_nonzeros=num_nonzeros,
    )


def solve_base_flow_utilization(
    network: NetworkData,
    demand_vector: np.ndarray,
    voll: float,
    spill_penalty: float,
    time_limit_s: float,
    threads: int,
    angle_bound_rad: float,
) -> np.ndarray:
    gp, GRB = require_gurobi()
    model = gp.Model("contingency_prescreen")
    empty_schedule = pd.DataFrame()
    state = add_operating_state(
        model,
        "peak_base",
        network,
        demand_vector[None, :],
        [],
        empty_schedule,
        None,
        1.0,
        angle_bound_rad,
        gp,
        GRB,
    )
    model.setObjective(
        state.generation_cost + voll * state.ens_mwh + spill_penalty * state.spilled_mwh,
        GRB.MINIMIZE,
    )
    apply_solver_parameters(model, time_limit_s, 0.0, threads, False, is_mip=False)
    model.optimize()
    if model.SolCount < 1:
        status = status_name(model.Status, GRB)
        model.dispose()
        raise RuntimeError(f"Contingency pre-screen dispatch failed with {status}")
    flow = np.array([state.flow[0, branch].X for branch in range(network.n_branch)])
    utilization = np.abs(flow) / np.maximum(network.branch_limit_mw, 1e-9)
    model.dispose()
    return utilization


def select_contingencies(
    network: NetworkData,
    base_demand: np.ndarray,
    count: int,
    include_generators: bool,
    voll: float,
    spill_penalty: float,
    time_limit_s: float,
    threads: int,
    angle_bound_rad: float,
) -> list[Contingency]:
    candidates = candidate_contingencies(network, include_generators)
    if count >= len(candidates):
        return candidates
    peak = base_demand[int(np.argmax(base_demand.sum(axis=1)))]
    utilization = solve_base_flow_utilization(
        network, peak, voll, spill_penalty, min(time_limit_s, 300.0), threads, angle_bound_rad
    )
    total_capacity = max(float(network.p_max_mw.sum()), 1e-9)

    def score(contingency: Contingency) -> float:
        if contingency.kind == "branch":
            return float(utilization[contingency.element])
        return float(network.p_max_mw[contingency.element] / total_capacity)

    return sorted(candidates, key=score, reverse=True)[:count]


def solve_evaluation_requests(
    name: str,
    network: NetworkData,
    requests: Sequence[EvaluationRequest],
    tasks: Sequence[OutageTask],
    schedule: pd.DataFrame,
    step_hours: float,
    voll: float,
    spill_penalty: float,
    time_limit_s: float,
    threads: int,
    angle_bound_rad: float,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    gp, GRB = require_gurobi()
    model = gp.Model(sanitize(name))
    handles: list[tuple[EvaluationRequest, StateHandle]] = []
    objective = gp.QuadExpr()
    for index, request in enumerate(requests):
        handle = add_operating_state(
            model,
            f"{request.label}_{index}",
            network,
            request.demand,
            tasks,
            schedule,
            request.contingency,
            step_hours,
            angle_bound_rad,
            gp,
            GRB,
        )
        handles.append((request, handle))
        objective += handle.generation_cost + voll * handle.ens_mwh + spill_penalty * handle.spilled_mwh
    model.setObjective(objective, GRB.MINIMIZE)
    apply_solver_parameters(model, time_limit_s, 0.0, threads, False, is_mip=False)
    if np.any(network.cost_quadratic < -1e-12):
        model.Params.NonConvex = 2
    model.optimize()
    if model.SolCount < 1:
        status = status_name(model.Status, GRB)
        model.dispose()
        raise RuntimeError(f"Evaluation batch {name} failed with {status}")

    records: list[dict[str, Any]] = []
    for request, handle in handles:
        records.append(
            {
                "scenario": request.scenario,
                "variant": request.variant,
                "contingency": request.contingency.name if request.contingency else "base",
                "operating_cost": float(handle.generation_cost.getValue()),
                "ens_mwh": float(handle.ens_mwh.getValue()),
                "spilled_mwh": float(handle.spilled_mwh.getValue()),
            }
        )
    model.dispose()
    return records


def evaluate_schedule(
    solution: ScheduleSolution,
    network: NetworkData,
    loads: np.ndarray,
    sampled_contingencies: Sequence[Optional[Contingency]],
    tasks: Sequence[OutageTask],
    step_hours: float,
    batch_size: int,
    voll: float,
    spill_penalty: float,
    time_limit_s: float,
    threads: int,
    angle_bound_rad: float,
) -> pd.DataFrame:
    if len(loads) != len(sampled_contingencies):
        raise ValueError("Test load and contingency sample counts do not match")
    all_records: list[dict[str, Any]] = []
    for batch_start in range(0, len(loads), batch_size):
        batch_stop = min(batch_start + batch_size, len(loads))
        requests: list[EvaluationRequest] = []
        for scenario in range(batch_start, batch_stop):
            requests.append(
                EvaluationRequest(
                    f"s{scenario}_normal", scenario, "normal", loads[scenario], None
                )
            )
            requests.append(
                EvaluationRequest(
                    f"s{scenario}_joint",
                    scenario,
                    "joint",
                    loads[scenario],
                    sampled_contingencies[scenario],
                )
            )
        all_records.extend(
            solve_evaluation_requests(
                f"eval_{solution.name}_{batch_start}",
                network,
                requests,
                tasks,
                solution.schedule,
                step_hours,
                voll,
                spill_penalty,
                time_limit_s,
                threads,
                angle_bound_rad,
            )
        )
        LOGGER.info("%s evaluation: %d/%d scenarios", solution.name, batch_stop, len(loads))

    raw = pd.DataFrame(all_records)
    normal = raw[raw["variant"] == "normal"].set_index("scenario")
    joint = raw[raw["variant"] == "joint"].set_index("scenario")
    result = pd.DataFrame(index=normal.index)
    result["scheduler"] = solution.name
    result["contingency"] = joint["contingency"]
    result["operating_cost_normal"] = normal["operating_cost"]
    result["operating_cost_joint"] = joint["operating_cost"]
    result["ens_normal_mwh"] = normal["ens_mwh"]
    result["ens_joint_mwh"] = joint["ens_mwh"]
    result["spilled_normal_mwh"] = normal["spilled_mwh"]
    result["spilled_joint_mwh"] = joint["spilled_mwh"]
    result["n1_violation"] = (
        (result["contingency"] != "base") & (result["ens_joint_mwh"] > 1e-6)
    )
    return result.reset_index()


def evaluate_n1_envelope(
    solution: ScheduleSolution,
    network: NetworkData,
    loads: np.ndarray,
    contingencies: Sequence[Contingency],
    tasks: Sequence[OutageTask],
    step_hours: float,
    batch_size: int,
    voll: float,
    spill_penalty: float,
    time_limit_s: float,
    threads: int,
    angle_bound_rad: float,
) -> pd.DataFrame:
    if len(loads) == 0 or not contingencies:
        return pd.DataFrame(
            columns=["scenario", "max_n1_ens_mwh", "mean_n1_ens_mwh", "n1_violation_fraction"]
        )
    requests = [
        EvaluationRequest(
            f"env_s{scenario}_{contingency.name}",
            scenario,
            contingency.name,
            loads[scenario],
            contingency,
        )
        for scenario in range(len(loads))
        for contingency in contingencies
    ]
    records: list[dict[str, Any]] = []
    for start in range(0, len(requests), batch_size):
        records.extend(
            solve_evaluation_requests(
                f"envelope_{solution.name}_{start}",
                network,
                requests[start : start + batch_size],
                tasks,
                solution.schedule,
                step_hours,
                voll,
                spill_penalty,
                time_limit_s,
                threads,
                angle_bound_rad,
            )
        )
    raw = pd.DataFrame(records)
    grouped = raw.groupby("scenario")["ens_mwh"]
    envelope = grouped.agg(max_n1_ens_mwh="max", mean_n1_ens_mwh="mean").reset_index()
    violation = raw.assign(violation=raw["ens_mwh"] > 1e-6).groupby("scenario")["violation"].mean()
    envelope["n1_violation_fraction"] = envelope["scenario"].map(violation)
    envelope["scheduler"] = solution.name
    return envelope


def summarize_evaluation(
    solution: ScheduleSolution,
    evaluation: pd.DataFrame,
    envelope: pd.DataFrame,
    alpha: float,
) -> dict[str, Any]:
    var, cvar = empirical_var_cvar(evaluation["ens_joint_mwh"], alpha)
    contingency_rows = evaluation[evaluation["contingency"] != "base"]
    record: dict[str, Any] = {
        "scheduler": solution.name,
        "status": solution.status,
        "expected_operating_cost_normal": evaluation["operating_cost_normal"].mean(),
        "expected_operating_cost_joint": evaluation["operating_cost_joint"].mean(),
        "maintenance_cost": solution.maintenance_cost,
        "mean_normal_ens_mwh": evaluation["ens_normal_mwh"].mean(),
        "mean_joint_ens_mwh": evaluation["ens_joint_mwh"].mean(),
        "conditional_n1_ens_mwh": (
            contingency_rows["ens_joint_mwh"].mean() if len(contingency_rows) else np.nan
        ),
        f"var_{int(round(100 * alpha))}_ens_mwh": var,
        f"cvar_{int(round(100 * alpha))}_ens_mwh": cvar,
        "probability_any_ens": (evaluation["ens_joint_mwh"] > 1e-6).mean(),
        "maximum_joint_ens_mwh": evaluation["ens_joint_mwh"].max(),
        "sampled_n1_violation_rate": evaluation["n1_violation"].mean(),
        "training_cvar_mwh": solution.training_cvar_mwh,
        "priority_delay_score": solution.priority_delay_score,
        "build_time_s": solution.build_time_s,
        "solve_time_s": solution.solve_time_s,
        "mip_gap": solution.mip_gap,
        "num_variables": solution.num_variables,
        "num_constraints": solution.num_constraints,
        "num_nonzeros": solution.num_nonzeros,
    }
    if len(envelope):
        record.update(
            {
                "mean_max_n1_ens_mwh": envelope["max_n1_ens_mwh"].mean(),
                "worst_n1_ens_mwh": envelope["max_n1_ens_mwh"].max(),
                "all_n1_violation_fraction": envelope["n1_violation_fraction"].mean(),
            }
        )
    else:
        record.update(
            {
                "mean_max_n1_ens_mwh": np.nan,
                "worst_n1_ens_mwh": np.nan,
                "all_n1_violation_fraction": np.nan,
            }
        )
    return record


def comparison_statistics(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    alpha: float,
    seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    base_cost = baseline["operating_cost_normal"].to_numpy()
    candidate_cost = candidate["operating_cost_normal"].to_numpy()
    base_loss = baseline["ens_joint_mwh"].to_numpy()
    candidate_loss = candidate["ens_joint_mwh"].to_numpy()
    cost_delta, cost_low, cost_high = paired_bootstrap_interval(
        base_cost, candidate_cost, "mean", alpha, seed, bootstrap_repetitions
    )
    cvar_delta, cvar_low, cvar_high = paired_bootstrap_interval(
        base_loss, candidate_loss, "cvar", alpha, seed + 1, bootstrap_repetitions
    )
    base_cvar = empirical_var_cvar(base_loss, alpha)[1]
    candidate_cvar = empirical_var_cvar(candidate_loss, alpha)[1]
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "cost_delta": cost_delta,
        "cost_delta_percent": 100.0 * cost_delta / max(abs(base_cost.mean()), 1e-9),
        "cost_delta_ci95_low": cost_low,
        "cost_delta_ci95_high": cost_high,
        "cvar_delta_mwh": cvar_delta,
        "cvar_delta_percent": 100.0 * cvar_delta / max(abs(base_cvar), 1e-9),
        "cvar_delta_ci95_low": cvar_low,
        "cvar_delta_ci95_high": cvar_high,
        "baseline_cvar_mwh": base_cvar,
        "candidate_cvar_mwh": candidate_cvar,
        "lower_cost_point_estimate": bool(cost_delta < 0),
        "risk_not_higher_point_estimate": bool(candidate_cvar <= base_cvar),
        "strict_paired_claim_supported": bool(cost_high < 0 and cvar_high <= 0),
    }


def infer_config_path(root: Path, system: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else root / explicit
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    exact_candidates = (
        root / "config" / f"{system}_scheduler.json",
        root / "config" / "wp3_optim" / f"{system}_scheduler.json",
    )
    for path in exact_candidates:
        if path.is_file():
            return path
    matches = sorted(
        path
        for path in (root / "config").rglob("*.json")
        if system.lower() in path.name.lower() and "sched" in path.name.lower()
    ) if (root / "config").is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RuntimeError(
            f"Several scheduler configurations match {system}: "
            + ", ".join(str(path) for path in matches)
            + ". Select one with --config."
        )
    raise FileNotFoundError(
        f"No scheduler configuration found for {system}; pass --config explicitly."
    )


def resolved_settings(args: argparse.Namespace) -> ExperimentSettings:
    settings = PROFILES[args.profile]
    updates: dict[str, Any] = {}
    mappings = {
        "resolution": "resolution",
        "horizon_steps": "horizon_steps",
        "outage_preset": "planned_outage_preset",
        "max_concurrent": "max_concurrent_outages",
        "contingencies": "n_contingencies",
        "train_scenarios": "n_train",
        "validation_scenarios": "n_validation",
        "test_scenarios": "n_test",
        "alpha": "alpha",
        "risk_budget_factors": "risk_budget_factors",
        "time_limit": "time_limit_s",
        "mip_gap": "mip_gap",
        "evaluation_batch_size": "evaluation_batch_size",
        "n1_envelope_scenarios": "n_n1_envelope_scenarios",
    }
    for argument, field in mappings.items():
        value = getattr(args, argument)
        if value is not None:
            updates[field] = value
    return replace(settings, **updates)


def load_or_generate_scenarios(
    path: Optional[Path],
    expected_count: int,
    base: np.ndarray,
    history: np.ndarray,
    family: str,
    seed: int,
    resolution: str,
    student_df: float,
    block_length: int,
    residual_scale: float,
) -> np.ndarray:
    loaded = load_scenario_array(path, tuple(base.shape))
    if loaded is not None:
        if len(loaded) < expected_count:
            raise ValueError(f"{path} contains {len(loaded)} scenarios; {expected_count} are required")
        return loaded[:expected_count]
    return generate_load_scenarios(
        base,
        history,
        expected_count,
        family,
        seed,
        resolution,
        student_df,
        block_length,
        residual_scale,
    )


def solution_stat_record(solution: ScheduleSolution, family: str, size: int) -> dict[str, Any]:
    return {
        "experiment": family,
        "size": size,
        "scheduler": solution.name,
        "status": solution.status,
        "objective": solution.objective,
        "best_bound": solution.best_bound,
        "mip_gap": solution.mip_gap,
        "build_time_s": solution.build_time_s,
        "solve_time_s": solution.solve_time_s,
        "num_variables": solution.num_variables,
        "num_constraints": solution.num_constraints,
        "num_nonzeros": solution.num_nonzeros,
        "training_cvar_mwh": solution.training_cvar_mwh,
    }


def run_scalability_analysis(
    args: argparse.Namespace,
    settings: ExperimentSettings,
    network: NetworkData,
    base_demand: np.ndarray,
    tasks: Sequence[OutageTask],
    all_selected_contingencies: Sequence[Contingency],
    base_probability: float,
    train_loads: np.ndarray,
    step_hours: float,
    risk_budget_mwh: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for n_scenarios in args.scalability_scenarios:
        if n_scenarios > len(train_loads):
            LOGGER.warning("Skipping scalability S=%d; only %d scenarios exist", n_scenarios, len(train_loads))
            continue
        sampled = sample_joint_contingencies(
            all_selected_contingencies[: settings.n_contingencies],
            base_probability,
            n_scenarios,
            args.seed + 7000 + n_scenarios,
        )
        try:
            solution = solve_schedule_model(
                name=f"scale_risk_S{n_scenarios}",
                mode="risk_equivalent",
                network=network,
                base_demand=base_demand,
                tasks=tasks,
                hard_contingencies=all_selected_contingencies[: settings.n_contingencies],
                risk_demands=train_loads[:n_scenarios],
                risk_contingencies=sampled,
                step_hours=step_hours,
                max_concurrent=settings.max_concurrent_outages,
                alpha=settings.alpha,
                risk_budget_mwh=risk_budget_mwh,
                risk_weight=0.0,
                priority_weight=args.priority_weight,
                voll=args.voll,
                spill_penalty=args.spill_penalty,
                strict_n1=args.strict_n1,
                shed_tolerance_mwh=args.shed_tolerance_mwh,
                must_protect_count=args.must_protect_count,
                time_limit_s=settings.time_limit_s,
                mip_gap=settings.mip_gap,
                threads=args.threads,
                solver_log=args.solver_log,
                angle_bound_rad=args.angle_bound,
            )
            records.append(solution_stat_record(solution, "training_scenarios", n_scenarios))
        except RuntimeError as exc:
            records.append(
                {
                    "experiment": "training_scenarios",
                    "size": n_scenarios,
                    "scheduler": f"scale_risk_S{n_scenarios}",
                    "status": f"FAILED: {exc}",
                }
            )

    for n_contingencies in args.scalability_contingencies:
        if n_contingencies > len(all_selected_contingencies):
            LOGGER.warning(
                "Skipping scalability C=%d; only %d contingencies were selected",
                n_contingencies,
                len(all_selected_contingencies),
            )
            continue
        try:
            solution = solve_schedule_model(
                name=f"scale_det_C{n_contingencies}",
                mode="deterministic",
                network=network,
                base_demand=base_demand,
                tasks=tasks,
                hard_contingencies=all_selected_contingencies[:n_contingencies],
                risk_demands=None,
                risk_contingencies=None,
                step_hours=step_hours,
                max_concurrent=settings.max_concurrent_outages,
                alpha=settings.alpha,
                risk_budget_mwh=None,
                risk_weight=0.0,
                priority_weight=args.priority_weight,
                voll=args.voll,
                spill_penalty=args.spill_penalty,
                strict_n1=args.strict_n1,
                shed_tolerance_mwh=args.shed_tolerance_mwh,
                must_protect_count=0,
                time_limit_s=settings.time_limit_s,
                mip_gap=settings.mip_gap,
                threads=args.threads,
                solver_log=args.solver_log,
                angle_bound_rad=args.angle_bound,
            )
            records.append(solution_stat_record(solution, "hard_contingencies", n_contingencies))
        except RuntimeError as exc:
            records.append(
                {
                    "experiment": "hard_contingencies",
                    "size": n_contingencies,
                    "scheduler": f"scale_det_C{n_contingencies}",
                    "status": f"FAILED: {exc}",
                }
            )
    return pd.DataFrame(records)


def plot_results(
    output_dir: Path,
    solutions: Sequence[ScheduleSolution],
    evaluations: pd.DataFrame,
    summary: pd.DataFrame,
    scalability: pd.DataFrame,
    alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Schedule heat maps.
    figure, axes = plt.subplots(
        len(solutions),
        1,
        figsize=(12, max(2.8, 2.3 * len(solutions))),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, solution in zip(axes[:, 0], solutions):
        image = axis.imshow(solution.schedule.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axis.set_yticks(range(len(solution.schedule.index)), solution.schedule.index)
        axis.set_ylabel(solution.name)
        axis.set_xlabel("Planning step")
        axis.set_xticks(range(0, solution.schedule.shape[1], max(1, solution.schedule.shape[1] // 12)))
    figure.colorbar(image, ax=axes[:, 0], label="Outage active", shrink=0.7)
    figure.savefig(output_dir / "schedule_comparison.png", dpi=180)
    plt.close(figure)

    # Empirical loss distributions on the common test set.
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for scheduler, group in evaluations.groupby("scheduler", sort=False):
        values = np.sort(group["ens_joint_mwh"].to_numpy())
        probability = np.arange(1, len(values) + 1) / len(values)
        axis.step(values, probability, where="post", label=scheduler)
    axis.set_xlabel("Joint-scenario ENS [MWh]")
    axis.set_ylabel("Empirical cumulative probability")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "ens_empirical_cdf.png", dpi=180)
    plt.close(figure)

    # Cost-risk plane.
    cvar_column = f"cvar_{int(round(100 * alpha))}_ens_mwh"
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(summary["expected_operating_cost_normal"], summary[cvar_column], s=70)
    for _, row in summary.iterrows():
        axis.annotate(
            row["scheduler"],
            (row["expected_operating_cost_normal"], row[cvar_column]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Expected normal operating cost [case cost units]")
    axis.set_ylabel(f"CVaR$_{{{alpha:.2f}}}$(ENS) [MWh]")
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "cost_risk_comparison.png", dpi=180)
    plt.close(figure)

    if len(scalability) and "solve_time_s" in scalability:
        valid = scalability[pd.to_numeric(scalability["solve_time_s"], errors="coerce").notna()]
        if len(valid):
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for experiment, group in valid.groupby("experiment"):
                group = group.sort_values("size")
                axes[0].plot(group["size"], group["solve_time_s"], marker="o", label=experiment)
                axes[1].plot(group["size"], group["num_variables"], marker="o", label=experiment)
            axes[0].set_xlabel("Scaled dimension")
            axes[0].set_ylabel("Solve time [s]")
            axes[1].set_xlabel("Scaled dimension")
            axes[1].set_ylabel("Number of variables")
            for axis in axes:
                axis.grid(alpha=0.25)
                axis.legend()
            figure.savefig(output_dir / "scalability.png", dpi=180)
            plt.close(figure)


def write_report_table(summary: pd.DataFrame, output_path: Path, alpha: float) -> None:
    cvar = f"cvar_{int(round(100 * alpha))}_ens_mwh"
    var = f"var_{int(round(100 * alpha))}_ens_mwh"
    columns = [
        "scheduler",
        "expected_operating_cost_normal",
        "mean_joint_ens_mwh",
        var,
        cvar,
        "probability_any_ens",
        "worst_n1_ens_mwh",
        "solve_time_s",
        "mip_gap",
    ]
    names = {
        "scheduler": "Scheduler",
        "expected_operating_cost_normal": "Expected operating cost",
        "mean_joint_ens_mwh": "EENS [MWh]",
        var: f"VaR$_{{{alpha:.2f}}}$ [MWh]",
        cvar: f"CVaR$_{{{alpha:.2f}}}$ [MWh]",
        "probability_any_ens": "P(ENS>0)",
        "worst_n1_ens_mwh": "Worst N-1 ENS [MWh]",
        "solve_time_s": "Solve time [s]",
        "mip_gap": "MIP gap",
    }
    table = summary[columns].rename(columns=names)
    table["Scheduler"] = table["Scheduler"].str.replace("_", " ", regex=False)
    latex = table.to_latex(
        index=False,
        float_format=lambda value: f"{value:.3g}",
        caption="Out-of-sample comparison of deterministic and risk-informed outage schedules.",
        label="tab:deterministic_vs_risk_informed",
        position="htbp",
        escape=False,
    )
    output_path.write_text(latex, encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return repr(value)


def run_experiment(args: argparse.Namespace) -> Path:
    settings = resolved_settings(args)
    root = repository_root()
    config_path = infer_config_path(root, args.system, args.config)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else root / "output" / "wp3_scheduler_evaluation" / f"{args.system}_{settings.profile}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    network_object, hourly_frame, repository_config = load_repository_case(config_path)
    network = extract_network_data(network_object, args.default_branch_limit)
    hourly_internal = map_demand_to_internal_buses(network_object, hourly_frame, network.n_bus)
    aggregated, step_hours = aggregate_demand(hourly_internal, settings.resolution, args.representative)
    base_demand, base_start = select_window(aggregated, settings.horizon_steps, args.window)
    tasks = load_tasks(
        args.outage_plan_json,
        settings.planned_outage_preset,
        network.n_branch,
        network.n_gen,
    )
    tasks = remap_default_generator_tasks(
        tasks,
        network_object,
        custom_plan_supplied=args.outage_plan_json is not None,
        n_internal_generators=network.n_gen,
    )
    validate_tasks(
        tasks,
        settings.horizon_steps,
        step_hours,
        settings.max_concurrent_outages,
    )

    maximum_contingencies = max(
        settings.n_contingencies,
        max(args.scalability_contingencies) if args.run_scalability else settings.n_contingencies,
    )
    prescreened = select_contingencies(
        network,
        base_demand,
        maximum_contingencies,
        args.include_generator_contingencies,
        args.voll,
        args.spill_penalty,
        settings.time_limit_s,
        args.threads,
        args.angle_bound,
    )
    selected_raw = prescreened[: settings.n_contingencies]
    probability_input = load_contingency_probabilities(args.contingency_probability_csv)
    selected, base_probability, probability_interpretation = assign_probabilities(
        selected_raw, probability_input, args.no_contingency_probability
    )
    # Assign probabilities to the larger scalability list using the same rule.
    all_selected, scale_base_probability, _ = assign_probabilities(
        prescreened, probability_input, args.no_contingency_probability
    )

    max_train = max(
        settings.n_train,
        max(args.scalability_scenarios) if args.run_scalability else settings.n_train,
    )
    train_loads = load_or_generate_scenarios(
        args.train_scenarios_file,
        max_train,
        base_demand,
        aggregated,
        args.scenario_family,
        args.seed,
        settings.resolution,
        args.student_df,
        args.block_length,
        args.residual_scale,
    )
    validation_loads = load_or_generate_scenarios(
        args.validation_scenarios_file,
        settings.n_validation,
        base_demand,
        aggregated,
        args.scenario_family,
        args.seed + 1,
        settings.resolution,
        args.student_df,
        args.block_length,
        args.residual_scale,
    )
    test_loads = load_or_generate_scenarios(
        args.test_scenarios_file,
        settings.n_test,
        base_demand,
        aggregated,
        args.scenario_family,
        args.seed + 2,
        settings.resolution,
        args.student_df,
        args.block_length,
        args.residual_scale,
    )
    train_joint = sample_joint_contingencies(
        selected, base_probability, settings.n_train, args.seed + 100
    )
    validation_joint = sample_joint_contingencies(
        selected, base_probability, settings.n_validation, args.seed + 101
    )
    test_joint = sample_joint_contingencies(
        selected, base_probability, settings.n_test, args.seed + 102
    )

    tail_observations = (1.0 - settings.alpha) * settings.n_train
    if tail_observations < 2.0:
        LOGGER.warning(
            "Training CVaR uses only %.2f effective tail observations. "
            "Use alpha=0.90 or increase --train-scenarios.",
            tail_observations,
        )

    LOGGER.warning("Solving deterministic N-1 schedule (%s profile)", settings.profile)
    deterministic = solve_schedule_model(
        name="deterministic_n1",
        mode="deterministic",
        network=network,
        base_demand=base_demand,
        tasks=tasks,
        hard_contingencies=selected,
        risk_demands=None,
        risk_contingencies=None,
        step_hours=step_hours,
        max_concurrent=settings.max_concurrent_outages,
        alpha=settings.alpha,
        risk_budget_mwh=None,
        risk_weight=0.0,
        priority_weight=args.priority_weight,
        voll=args.voll,
        spill_penalty=args.spill_penalty,
        strict_n1=args.strict_n1,
        shed_tolerance_mwh=args.shed_tolerance_mwh,
        must_protect_count=0,
        time_limit_s=settings.time_limit_s,
        mip_gap=settings.mip_gap,
        threads=args.threads,
        solver_log=args.solver_log,
        angle_bound_rad=args.angle_bound,
    )
    deterministic_validation = evaluate_schedule(
        deterministic,
        network,
        validation_loads,
        validation_joint,
        tasks,
        step_hours,
        settings.evaluation_batch_size,
        args.voll,
        args.spill_penalty,
        args.evaluation_time_limit,
        args.threads,
        args.angle_bound,
    )
    _, deterministic_validation_cvar = empirical_var_cvar(
        deterministic_validation["ens_joint_mwh"], settings.alpha
    )

    tuning_records: list[dict[str, Any]] = []
    risk_candidates: list[tuple[ScheduleSolution, pd.DataFrame]] = []
    for factor in settings.risk_budget_factors:
        budget = max(args.risk_budget_floor_mwh, factor * deterministic_validation_cvar)
        name = f"risk_equivalent_f{factor:g}"
        LOGGER.warning("Solving %s with CVaR budget %.6g MWh", name, budget)
        try:
            solution = solve_schedule_model(
                name=name,
                mode="risk_equivalent",
                network=network,
                base_demand=base_demand,
                tasks=tasks,
                hard_contingencies=selected,
                risk_demands=train_loads[: settings.n_train],
                risk_contingencies=train_joint,
                step_hours=step_hours,
                max_concurrent=settings.max_concurrent_outages,
                alpha=settings.alpha,
                risk_budget_mwh=budget,
                risk_weight=0.0,
                priority_weight=args.priority_weight,
                voll=args.voll,
                spill_penalty=args.spill_penalty,
                strict_n1=args.strict_n1,
                shed_tolerance_mwh=args.shed_tolerance_mwh,
                must_protect_count=args.must_protect_count,
                time_limit_s=settings.time_limit_s,
                mip_gap=settings.mip_gap,
                threads=args.threads,
                solver_log=args.solver_log,
                angle_bound_rad=args.angle_bound,
            )
            validation = evaluate_schedule(
                solution,
                network,
                validation_loads,
                validation_joint,
                tasks,
                step_hours,
                settings.evaluation_batch_size,
                args.voll,
                args.spill_penalty,
                args.evaluation_time_limit,
                args.threads,
                args.angle_bound,
            )
            validation_cvar = empirical_var_cvar(validation["ens_joint_mwh"], settings.alpha)[1]
            tuning_records.append(
                {
                    "factor": factor,
                    "risk_budget_mwh": budget,
                    "status": solution.status,
                    "validation_operating_cost": validation["operating_cost_normal"].mean(),
                    "validation_mean_ens_mwh": validation["ens_joint_mwh"].mean(),
                    "validation_cvar_mwh": validation_cvar,
                    "validation_risk_ratio_to_deterministic": (
                        validation_cvar / deterministic_validation_cvar
                        if deterministic_validation_cvar > 1e-12
                        else (0.0 if validation_cvar <= 1e-12 else np.inf)
                    ),
                    "solve_time_s": solution.solve_time_s,
                    "mip_gap": solution.mip_gap,
                }
            )
            risk_candidates.append((solution, validation))
        except RuntimeError as exc:
            tuning_records.append(
                {
                    "factor": factor,
                    "risk_budget_mwh": budget,
                    "status": f"FAILED: {exc}",
                }
            )

    if not risk_candidates:
        raise RuntimeError("No risk-equivalent candidate produced an incumbent schedule")
    risk_threshold = (
        deterministic_validation_cvar * (1.0 + args.risk_equivalence_tolerance)
        + args.risk_equivalence_absolute_tolerance_mwh
    )
    eligible = [
        item
        for item in risk_candidates
        if empirical_var_cvar(item[1]["ens_joint_mwh"], settings.alpha)[1] <= risk_threshold
    ]
    if eligible:
        selected_risk, _ = min(
            eligible, key=lambda item: float(item[1]["operating_cost_normal"].mean())
        )
        validation_risk_target_met = True
    else:
        selected_risk, _ = min(
            risk_candidates,
            key=lambda item: empirical_var_cvar(
                item[1]["ens_joint_mwh"], settings.alpha
            )[1],
        )
        validation_risk_target_met = False
        LOGGER.warning(
            "No candidate met the validation risk-equivalence threshold; "
            "selecting the candidate with the lowest validation CVaR."
        )

    solutions: list[ScheduleSolution] = [deterministic, selected_risk]
    if args.include_hybrid:
        LOGGER.warning("Solving hybrid N-1 + CVaR schedule")
        hybrid = solve_schedule_model(
            name="hybrid_n1_plus_cvar",
            mode="hybrid",
            network=network,
            base_demand=base_demand,
            tasks=tasks,
            hard_contingencies=selected,
            risk_demands=train_loads[: settings.n_train],
            risk_contingencies=[None] * settings.n_train,
            step_hours=step_hours,
            max_concurrent=settings.max_concurrent_outages,
            alpha=settings.alpha,
            risk_budget_mwh=None,
            risk_weight=args.hybrid_risk_weight,
            priority_weight=args.priority_weight,
            voll=args.voll,
            spill_penalty=args.spill_penalty,
            strict_n1=args.strict_n1,
            shed_tolerance_mwh=args.shed_tolerance_mwh,
            must_protect_count=0,
            time_limit_s=settings.time_limit_s,
            mip_gap=settings.mip_gap,
            threads=args.threads,
            solver_log=args.solver_log,
            angle_bound_rad=args.angle_bound,
        )
        solutions.append(hybrid)

    evaluation_tables: list[pd.DataFrame] = []
    envelope_tables: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    for solution in solutions:
        LOGGER.warning("Out-of-sample evaluation: %s", solution.name)
        evaluation = evaluate_schedule(
            solution,
            network,
            test_loads,
            test_joint,
            tasks,
            step_hours,
            settings.evaluation_batch_size,
            args.voll,
            args.spill_penalty,
            args.evaluation_time_limit,
            args.threads,
            args.angle_bound,
        )
        envelope = evaluate_n1_envelope(
            solution,
            network,
            test_loads[: settings.n_n1_envelope_scenarios],
            selected,
            tasks,
            step_hours,
            settings.evaluation_batch_size,
            args.voll,
            args.spill_penalty,
            args.evaluation_time_limit,
            args.threads,
            args.angle_bound,
        )
        evaluation_tables.append(evaluation)
        envelope_tables.append(envelope)
        summary_records.append(summarize_evaluation(solution, evaluation, envelope, settings.alpha))

    evaluations = pd.concat(evaluation_tables, ignore_index=True)
    envelopes = pd.concat(envelope_tables, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    baseline_evaluation = evaluation_tables[0]
    comparisons = pd.DataFrame(
        [
            comparison_statistics(
                baseline_evaluation,
                evaluation,
                deterministic.name,
                solution.name,
                settings.alpha,
                args.seed + 500 + index,
                args.bootstrap_repetitions,
            )
            for index, (solution, evaluation) in enumerate(
                zip(solutions[1:], evaluation_tables[1:]), start=1
            )
        ]
    )

    scalability = pd.DataFrame()
    if args.run_scalability:
        scalability = run_scalability_analysis(
            args,
            settings,
            network,
            base_demand,
            tasks,
            all_selected,
            scale_base_probability,
            train_loads,
            step_hours,
            max(args.risk_budget_floor_mwh, deterministic_validation_cvar),
        )

    # Persist numerical outputs.
    pd.DataFrame([asdict(task) for task in tasks]).to_csv(output_dir / "planned_outages.csv", index=False)
    pd.DataFrame([asdict(item) for item in selected]).to_csv(
        output_dir / "selected_contingencies.csv", index=False
    )
    for solution in solutions:
        solution.schedule.to_csv(output_dir / f"schedule_{sanitize(solution.name)}.csv")
        solution.starts.to_csv(output_dir / f"starts_{sanitize(solution.name)}.csv", index=False)
    pd.DataFrame(tuning_records).to_csv(output_dir / "risk_budget_tuning.csv", index=False)
    evaluations.to_csv(output_dir / "out_of_sample_scenarios.csv", index=False)
    envelopes.to_csv(output_dir / "n1_envelope_evaluation.csv", index=False)
    summary.to_csv(output_dir / "kpi_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    scalability.to_csv(output_dir / "scalability.csv", index=False)
    write_report_table(summary, output_dir / "kpi_summary.tex", settings.alpha)
    plot_results(output_dir, solutions, evaluations, summary, scalability, settings.alpha)

    if args.save_scenarios:
        np.savez_compressed(
            output_dir / "scenario_sets.npz",
            base=base_demand,
            train=train_loads[: settings.n_train],
            validation=validation_loads,
            test=test_loads,
        )

    manifest = {
        "system": args.system,
        "profile": asdict(settings),
        "config_path": config_path,
        "repository_config": repository_config,
        "base_window_start_aggregated_index": base_start,
        "step_hours": step_hours,
        "representative": args.representative,
        "scenario_family": args.scenario_family,
        "scenario_seed": args.seed,
        "student_df": args.student_df,
        "residual_scale": args.residual_scale,
        "cost_source": network.cost_source,
        "contingency_probability_interpretation": probability_interpretation,
        "base_probability": base_probability,
        "strict_n1": args.strict_n1,
        "must_protect_count_in_risk_model": args.must_protect_count,
        "selected_risk_scheduler": selected_risk.name,
        "selected_risk_scheduler_met_validation_target": validation_risk_target_met,
        "deterministic_validation_cvar_mwh": deterministic_validation_cvar,
        "risk_selection_threshold_mwh": risk_threshold,
        "warnings": [
            "Uniform contingency weights are a stress-test assumption, not empirical failure probabilities."
            if not probability_input
            else "Contingency probabilities were normalized over the selected set.",
            "Aggregated representative loads omit within-step chronology; validate final schedules at hourly resolution.",
            "Maintenance cost is constant when every listed task is mandatory; cost differences therefore arise from dispatch.",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )

    print("\nOut-of-sample KPI summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print("\nPaired comparisons against deterministic N-1")
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print(f"\nResults written to: {output_dir.resolve()}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Full deterministic N-1 versus CVaR outage-scheduling evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--system", default="IEEE24")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resolution", choices=("H", "D", "W"), default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--window", choices=("peak", "first", "latest"), default="peak")
    parser.add_argument("--representative", choices=("mean", "system_peak"), default="mean")
    parser.add_argument("--outage-preset", choices=("smoke", "pilot", "full"), default=None)
    parser.add_argument("--outage-plan-json", type=Path, default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--contingencies", type=int, default=None)
    parser.add_argument(
        "--include-generator-contingencies",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--contingency-probability-csv", type=Path, default=None)
    parser.add_argument("--no-contingency-probability", type=float, default=0.5)
    parser.add_argument("--must-protect-count", type=int, default=0)

    parser.add_argument("--scenario-family", choices=("empirical", "student_t"), default="empirical")
    parser.add_argument("--train-scenarios", type=int, default=None)
    parser.add_argument("--validation-scenarios", type=int, default=None)
    parser.add_argument("--test-scenarios", type=int, default=None)
    parser.add_argument("--train-scenarios-file", type=Path, default=None)
    parser.add_argument("--validation-scenarios-file", type=Path, default=None)
    parser.add_argument("--test-scenarios-file", type=Path, default=None)
    parser.add_argument("--student-df", type=float, default=3.0)
    parser.add_argument("--block-length", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--save-scenarios", action="store_true")

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--risk-budget-factors", type=parse_float_list, default=None)
    parser.add_argument("--risk-budget-floor-mwh", type=float, default=0.0)
    parser.add_argument("--risk-equivalence-tolerance", type=float, default=0.02)
    parser.add_argument("--risk-equivalence-absolute-tolerance-mwh", type=float, default=1e-6)
    parser.add_argument("--include-hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-risk-weight", type=float, default=1e5)
    parser.add_argument("--strict-n1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shed-tolerance-mwh", type=float, default=1e-5)
    parser.add_argument("--voll", type=float, default=1e6)
    parser.add_argument("--spill-penalty", type=float, default=1.0)
    parser.add_argument("--priority-weight", type=float, default=1e3)
    parser.add_argument("--angle-bound", type=float, default=math.pi)
    parser.add_argument("--default-branch-limit", type=float, default=500.0)

    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--evaluation-time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int, default=None)
    parser.add_argument("--n1-envelope-scenarios", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)

    parser.add_argument("--run-scalability", action="store_true")
    parser.add_argument(
        "--scalability-scenarios",
        type=parse_int_list,
        default=(5, 10, 25, 50),
    )
    parser.add_argument(
        "--scalability-contingencies",
        type=parse_int_list,
        default=(1, 5, 10, 25),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(_root=root)
    return parser


def validate_arguments(args: argparse.Namespace, settings: ExperimentSettings) -> None:
    positive = {
        "horizon_steps": settings.horizon_steps,
        "n_contingencies": settings.n_contingencies,
        "n_train": settings.n_train,
        "n_validation": settings.n_validation,
        "n_test": settings.n_test,
        "max_concurrent_outages": settings.max_concurrent_outages,
        "threads": args.threads,
        "evaluation_batch_size": settings.evaluation_batch_size,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("These parameters must be positive: " + ", ".join(invalid))
    if not 0.0 < settings.alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if args.student_df <= 2.0:
        raise ValueError("Student-t degrees of freedom must exceed two for finite variance")
    if any(value <= 0 for value in settings.risk_budget_factors):
        raise ValueError("Risk-budget factors must be positive")
    if args.must_protect_count < 0:
        raise ValueError("must-protect-count cannot be negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    settings = resolved_settings(args)
    validate_arguments(args, settings)
    try:
        run_experiment(args)
    except Exception as exc:
        LOGGER.exception("Evaluation failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
