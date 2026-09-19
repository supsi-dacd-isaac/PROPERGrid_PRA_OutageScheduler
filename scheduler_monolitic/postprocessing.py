"""Result extraction and persistence for the unified monolithic scheduler."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .risk_measures import summarise_losses

if TYPE_CHECKING:
    from .monolithic_scos import ModelArtifacts


@dataclass
class SCOSResult:
    formulation: str
    schedule: pd.DataFrame
    start_periods: pd.Series
    scenario_losses: pd.Series
    scenario_probabilities: pd.Series
    worst_dns: pd.DataFrame
    power_generated: pd.DataFrame
    line_flows: pd.DataFrame
    base_curtailment: pd.DataFrame
    contingency_curtailment: pd.DataFrame
    scenario_contingency_energy: pd.DataFrame
    risk_metrics: dict[str, float]
    maintenance_utility: float
    expected_generation: float
    solver_summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_solution: dict[str, float] | None = None

    def as_legacy_dictionary(self) -> dict[str, Any]:
        """Compatibility view expected by the previous plotting utilities."""
        return {
            "X_OutageSchedule": self.schedule,
            "PowerGenerated": self.power_generated,
            "Line_Flows": self.line_flows,
            "WC_CURTAIL": self.base_curtailment,
            "WC_CURTAIL_CON": self.contingency_curtailment,
            "GEN_PLUS_CURTAILED": (
                self.power_generated.sum(axis=0).to_numpy()
                + self.base_curtailment.sum(axis=0).to_numpy()
            ),
            "ScenarioLosses": self.scenario_losses,
            "ScenarioProbabilities": self.scenario_probabilities,
            "WorstDNS": self.worst_dns,
            "RiskMetrics": self.risk_metrics,
            "MaintenanceUtility": self.maintenance_utility,
            "ScenarioContingencyEnergy": self.scenario_contingency_energy,
        }


def _status_name(status: int) -> str:
    try:
        from gurobipy import GRB

        mapping = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }
        return mapping.get(status, str(status))
    except ImportError:  # pragma: no cover
        return str(status)


def _safe_model_attr(model: Any, attr_name: str, default: Any = None) -> Any:
    """Return a Gurobi model attribute without failing post-processing.

    Some attributes, notably ``MIPGap`` and objective-bound attributes, can be
    unavailable after multi-objective runs stopped by a time limit even when a
    feasible incumbent is present. Post-processing must still extract and save
    the incumbent solution in that case.
    """
    try:
        return model.getAttr(attr_name)
    except Exception:
        try:
            return getattr(model, attr_name)
        except Exception:
            return default


def _safe_float_model_attr(model: Any, attr_name: str, default: float = float("nan")) -> float:
    value = _safe_model_attr(model, attr_name, default)
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int_model_attr(model: Any, attr_name: str, default: int = 0) -> int:
    value = _safe_model_attr(model, attr_name, default)
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def extract_result(artifacts: "ModelArtifacts") -> SCOSResult:
    model = artifacts.model
    d = artifacts.data
    variables = artifacts.variables
    expressions = artifacts.expressions

    if model.SolCount < 1:
        raise RuntimeError(f"No feasible solution is available; Gurobi status is {_status_name(model.Status)}.")

    probabilities = pd.Series(d.scenario_probabilities, name="probability", dtype=float)
    schedule = pd.DataFrame(
        [
            [variables["x"][time, outage].X for time in d.times]
            for outage in d.outages
        ],
        index=d.outages,
        columns=d.times,
        dtype=float,
    )

    start_periods = {}
    for outage in d.outages:
        active = np.flatnonzero(schedule.loc[outage].to_numpy() > 0.5)
        start_periods[outage] = int(active[0]) if active.size else -1
    start_periods_series = pd.Series(start_periods, name="start_step", dtype=int)

    scenario_losses = pd.Series(
        {scenario: variables["scenario_loss"][scenario].X for scenario in d.scenarios},
        name="loss_mw_step",
        dtype=float,
    )
    worst_dns = pd.DataFrame(
        [
            [variables["worst_dns"][time, scenario].X for time in d.times]
            for scenario in d.scenarios
        ],
        index=d.scenarios,
        columns=d.times,
        dtype=float,
    )

    power_generated = pd.DataFrame(index=d.generators, columns=d.times, dtype=float)
    line_flows = pd.DataFrame(index=d.lines, columns=d.times, dtype=float)
    base_curtailment = pd.DataFrame(index=d.buses, columns=d.times, dtype=float)
    contingency_curtailment = pd.DataFrame(index=d.contingencies, columns=d.times, dtype=float)

    for time in d.times:
        for generator in d.generators:
            power_generated.loc[generator, time] = sum(
                d.scenario_probabilities[scenario]
                * variables["p"][time, generator, scenario].X
                for scenario in d.scenarios
            )
        for line in d.lines:
            line_flows.loc[line, time] = sum(
                d.scenario_probabilities[scenario]
                * variables["flow"][time, line, scenario].X
                for scenario in d.scenarios
            )
        for bus in d.buses:
            base_curtailment.loc[bus, time] = sum(
                d.scenario_probabilities[scenario]
                * variables["shed"][time, bus, scenario].X
                for scenario in d.scenarios
            )
        for contingency in d.contingencies:
            contingency_curtailment.loc[contingency, time] = sum(
                d.scenario_probabilities[scenario]
                * sum(
                    variables["shed_c"][time, bus, scenario, contingency].X
                    for bus in d.buses
                )
                for scenario in d.scenarios
            )

    scenario_contingency_energy = pd.DataFrame(
        index=d.scenarios,
        columns=d.contingencies,
        dtype=float,
    )
    for scenario in d.scenarios:
        for contingency in d.contingencies:
            scenario_contingency_energy.loc[scenario, contingency] = sum(
                d.time_weights[time]
                * sum(
                    variables["shed_c"][time, bus, scenario, contingency].X
                    for bus in d.buses
                )
                for time in d.times
            )

    metrics = summarise_losses(
        scenario_losses.to_numpy(),
        beta=artifacts.risk_config.beta,
        probabilities=probabilities.loc[scenario_losses.index].to_numpy(),
    ).to_dict()

    utility = float(expressions["maintenance_utility"].getValue())
    expected_generation = float(expressions["expected_generation"].getValue())
    is_mip = bool(_safe_int_model_attr(model, "IsMIP", 0))
    solver_summary = {
        "status": _status_name(_safe_int_model_attr(model, "Status", 0)),
        "status_code": _safe_int_model_attr(model, "Status", 0),
        "runtime_sec": _safe_float_model_attr(model, "Runtime"),
        "objective_value": _safe_float_model_attr(model, "ObjVal"),
        "objective_bound": _safe_float_model_attr(model, "ObjBound"),
        "mip_gap": _safe_float_model_attr(model, "MIPGap") if is_mip else 0.0,
        "node_count": _safe_float_model_attr(model, "NodeCount"),
        "iteration_count": _safe_float_model_attr(model, "IterCount"),
        "solution_count": _safe_int_model_attr(model, "SolCount", 0),
        "is_mip": is_mip,
    }

    raw_solution = None
    if artifacts.model_config.store_full_solution:
        raw_solution = {variable.VarName: float(variable.X) for variable in model.getVars()}

    metadata = {
        "beta": artifacts.risk_config.beta,
        "formulation": artifacts.risk_config.formulation,
        "n_scenarios": len(d.scenarios),
        "n_contingencies": len(d.contingencies),
        "contingencies": list(d.contingencies),
        "loss_definition": "sum_t weight_t * max(base DNS, N-1 contingency DNS)",
        "loss_unit": "MW-step",
        "scenario_dispatch": "scenario-adaptive preventive and corrective DC dispatch",
        "peak_demand_mw": d.peak_demand,
    }

    return SCOSResult(
        formulation=artifacts.risk_config.formulation,
        schedule=schedule,
        start_periods=start_periods_series,
        scenario_losses=scenario_losses,
        scenario_probabilities=probabilities,
        worst_dns=worst_dns,
        power_generated=power_generated,
        line_flows=line_flows,
        base_curtailment=base_curtailment,
        contingency_curtailment=contingency_curtailment,
        scenario_contingency_energy=scenario_contingency_energy,
        risk_metrics=metrics,
        maintenance_utility=utility,
        expected_generation=expected_generation,
        solver_summary=solver_summary,
        metadata=metadata,
        raw_solution=raw_solution,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_result_bundle(result: SCOSResult, output_dir: str | Path, run_name: str) -> Path:
    """Save a compact result bundle as pickle, JSON and CSV files."""
    directory = Path(output_dir).expanduser().resolve() / run_name
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / "result.pkl").open("wb") as stream:
        pickle.dump(result, stream)

    result.schedule.to_csv(directory / "outage_schedule.csv")
    result.start_periods.to_csv(directory / "outage_start_periods.csv")
    result.scenario_losses.to_csv(directory / "scenario_losses.csv")
    result.worst_dns.to_csv(directory / "scenario_worst_dns.csv")
    result.contingency_curtailment.to_csv(directory / "expected_contingency_curtailment.csv")
    result.scenario_contingency_energy.to_csv(directory / "scenario_contingency_energy.csv")

    summary = {
        "formulation": result.formulation,
        "risk_metrics": result.risk_metrics,
        "maintenance_utility": result.maintenance_utility,
        "expected_generation": result.expected_generation,
        "solver": result.solver_summary,
        "metadata": result.metadata,
    }
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(summary), stream, indent=2)

    if result.raw_solution is not None:
        with (directory / "raw_solution.json").open("w", encoding="utf-8") as stream:
            json.dump(result.raw_solution, stream)
    return directory


def load_result_bundle(path: str | Path) -> SCOSResult:
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / "result.pkl"
    with file_path.open("rb") as stream:
        result = pickle.load(stream)
    if not isinstance(result, SCOSResult):
        raise TypeError(f"Unexpected object in {file_path}: {type(result)!r}")
    return result
