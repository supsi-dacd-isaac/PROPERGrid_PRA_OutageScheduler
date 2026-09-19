"""Time-indexed Monte Carlo probabilistic risk assessment.

This module implements Algorithm 2 of the PROPER report. It keeps overload,
undervoltage, and overvoltage consequences separate. The optional composite
score is intended for ranking, not as a physical unit of system risk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from pra_psa.core.models import Contingency, normalize_contingencies


@dataclass(frozen=True)
class PRAConfig:
    """Numerical and severity settings for time-indexed PRA."""

    n_samples: int = 50
    step_hours: float = 1.0
    pf_mode: str = "ac"
    reference_dispatch: str = "opf"
    allow_pf_dispatch_fallback: bool = True
    loading_limit_percent: float = 100.0
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    voltage_weight: float = 100.0
    nonconvergence_penalty: float = 1_000.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2.")
        if self.step_hours <= 0:
            raise ValueError("step_hours must be positive.")
        if self.pf_mode not in {"ac", "dc"}:
            raise ValueError("pf_mode must be 'ac' or 'dc'.")
        if self.reference_dispatch not in {"opf", "pf"}:
            raise ValueError("reference_dispatch must be 'opf' or 'pf'.")
        if not 0 < self.v_min_pu < self.v_max_pu:
            raise ValueError("Require 0 < v_min_pu < v_max_pu.")


class OperatingStateSampler(Protocol):
    def sample(
        self,
        baseline_p_mw: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        *,
        time_index: Any,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class GaussianRelativeLoadSampler:
    """Compact correlated relative-error model for demonstrations.

    For report-quality studies, replace this with a fitted nodal forecasting
    model. The interface intentionally makes that substitution local.
    """

    common_sigma: float = 0.035
    nodal_sigma: float = 0.015
    lower_multiplier: float = 0.70
    upper_multiplier: float = 1.30

    def sample(
        self,
        baseline_p_mw: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        *,
        time_index: Any,
    ) -> np.ndarray:
        baseline = np.asarray(baseline_p_mw, dtype=float)
        common = rng.normal(0.0, self.common_sigma, size=(n_samples, 1))
        nodal = rng.normal(0.0, self.nodal_sigma, size=(n_samples, baseline.size))
        multiplier = np.clip(1.0 + common + nodal, self.lower_multiplier,self.upper_multiplier,)
        return np.maximum(baseline[None, :] * multiplier, 0.0)


class ContingencyProbabilityModel(Protocol):
    def probabilities(
        self,
        contingencies: Sequence[Contingency],
        *,
        operating_state_mw: np.ndarray,
        baseline_state_mw: np.ndarray,
        weather_regime: str | None,
        time_index: Any,
        horizon_hours: float,
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class UniformProbabilityModel:
    """Equal probability mass for the selected contingency catalogue."""

    total_contingency_probability: float = 0.01

    def probabilities(
        self,
        contingencies: Sequence[Contingency],
        **_: Any,
    ) -> Mapping[str, float]:
        if not 0 <= self.total_contingency_probability <= 1:
            raise ValueError("total_contingency_probability must be in [0, 1].")
        if not contingencies:
            return {}
        probability = self.total_contingency_probability / len(contingencies)
        return {item.contingency_id: probability for item in contingencies}


@dataclass(frozen=True)
class WeatherConditionedPoissonModel:
    """State- and weather-conditioned contingency occurrence model.

    Hourly rates are addressed first by ``element_type:index``, then by element
    type, and finally by ``default``. The state relationship is

    ``lambda(x)=lambda_0*m_weather*exp(load_beta*(sum(x)/sum(x_BL)-1))``.
    """

    rates_per_hour: Mapping[str, float]
    weather_multipliers: Mapping[str, float] | None = None
    load_beta: float = 0.0
    max_total_contingency_probability: float = 0.20

    def _rate(self, element_type: str, element_index: int) -> float:
        key = f"{element_type}:{element_index}"
        value = self.rates_per_hour.get(
            key,
            self.rates_per_hour.get(element_type, self.rates_per_hour.get("default", 0.0)),
        )
        value = float(value)
        if value < 0:
            raise ValueError(f"Negative failure rate for {key}.")
        return value

    def probabilities(
        self,
        contingencies: Sequence[Contingency],
        *,
        operating_state_mw: np.ndarray,
        baseline_state_mw: np.ndarray,
        weather_regime: str | None,
        horizon_hours: float,
        **_: Any,
    ) -> Mapping[str, float]:
        baseline_total = max(float(np.sum(baseline_state_mw)), 1e-12)
        relative_loading = float(np.sum(operating_state_mw)) / baseline_total - 1.0
        state_multiplier = float(np.exp(self.load_beta * relative_loading))
        weather_multiplier = 1.0
        if self.weather_multipliers is not None and weather_regime is not None:
            weather_multiplier = float(self.weather_multipliers.get(str(weather_regime), 1.0))
        if weather_multiplier < 0:
            raise ValueError("Weather hazard multipliers cannot be negative.")

        probabilities: dict[str, float] = {}
        for contingency in contingencies:
            probability = 1.0
            for outage in contingency.outages:
                rate = self._rate(outage.element_type, outage.element_index)
                rate *= weather_multiplier * state_multiplier
                probability *= 1.0 - np.exp(-rate * horizon_hours)
            probabilities[contingency.contingency_id] = float(probability)

        total = float(sum(probabilities.values()))
        if total > self.max_total_contingency_probability and total > 0:
            scale = self.max_total_contingency_probability / total
            probabilities = {key: value * scale for key, value in probabilities.items()}
        return probabilities


@dataclass
class TimeSeriesPRAResult:
    summary: dict[str, Any]
    risk_by_time: pd.DataFrame
    risk_by_contingency: pd.DataFrame
    risk_by_component: pd.DataFrame
    scenario_results: pd.DataFrame
    sampled_probabilities: pd.DataFrame

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(self.summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        self.risk_by_time.to_csv(output / "risk_by_time.csv", index=False)
        self.risk_by_contingency.to_csv(output / "risk_by_contingency.csv", index=False)
        self.risk_by_component.to_csv(output / "risk_by_component.csv", index=False)
        self.scenario_results.to_csv(output / "scenario_results.csv", index=False)
        self.sampled_probabilities.to_csv(output / "sampled_contingency_probabilities.csv", index=False)


def _run_pf(net: Any, mode: str) -> None:
    import pandapower as pp

    if mode == "ac":
        pp.runpp(
            net,
            calculate_voltage_angles=True,
            init="auto",
            enforce_q_lims=True,
            numba=False,
        )
    else:
        pp.rundcpp(net, numba=False)


def _run_reference_dispatch(net: Any, config: PRAConfig) -> str:
    """Solve OPF/PF and leave the resulting active dispatch on ``net``."""

    import pandapower as pp

    method = config.reference_dispatch
    try:
        if method == "opf":
            if config.pf_mode == "ac":
                pp.runopp(net, calculate_voltage_angles=True, init="pf", numba=False)
            else:
                pp.rundcopp(net, numba=False)
        else:
            _run_pf(net, config.pf_mode)
    except Exception:
        if method != "opf" or not config.allow_pf_dispatch_fallback:
            raise
        _run_pf(net, config.pf_mode)
        method = "pf_fallback"

    for table_name in ("gen", "sgen"):
        table = getattr(net, table_name, None)
        result = getattr(net, f"res_{table_name}", None)
        if table is None or result is None or table.empty or result.empty:
            continue
        if "p_mw" in result:
            table.loc[result.index, "p_mw"] = result["p_mw"].astype(float)
        if table_name == "gen" and "q_mvar" in result and "q_mvar" in table:
            table.loc[result.index, "q_mvar"] = result["q_mvar"].astype(float)
    return method


def _set_load(net: Any, p_mw: np.ndarray, q_over_p: np.ndarray) -> None:
    p = np.asarray(p_mw, dtype=float)
    if p.shape != (len(net.load),):
        raise ValueError(f"Expected {len(net.load)} load values, received {p.shape}.")
    net.load.loc[:, "p_mw"] = p
    if "q_mvar" in net.load:
        net.load.loc[:, "q_mvar"] = p * q_over_p


def _apply_contingency_in_place(net: Any, contingency: Contingency) -> None:
    for outage in contingency.outages:
        table = getattr(net, outage.element_type, None)
        if table is None or outage.element_index not in table.index:
            raise KeyError(f"Unknown outage {outage.element_type}:{outage.element_index}.")
        table.at[outage.element_index, "in_service"] = False


def _extract_severity(
    net: Any,
    config: PRAConfig,
    *,
    converged: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not converged:
        return {
            "overload_excess_percent": 0.0,
            "undervoltage_excess_pu": 0.0,
            "overvoltage_excess_pu": 0.0,
            "nonconvergence": 1.0,
            "max_loading_percent": float("nan"),
            "min_voltage_pu": float("nan"),
            "max_voltage_pu": float("nan"),
            "worst_performance_score": config.nonconvergence_penalty,
            "composite_severity": config.nonconvergence_penalty,
        }, []

    component_rows: list[dict[str, Any]] = []
    overload_total = 0.0
    max_loading = 0.0
    for element_type in ("line", "trafo"):
        table = getattr(net, element_type, None)
        result = getattr(net, f"res_{element_type}", None)
        if table is None or result is None or table.empty or result.empty:
            continue
        loading = pd.to_numeric(result.get("loading_percent"), errors="coerce")
        for element_index, value in loading.items():
            if not np.isfinite(value):
                continue
            excess = max(float(value) - config.loading_limit_percent, 0.0)
            overload_total += excess
            max_loading = max(max_loading, float(value))
            if excess > 0.0:
                component_rows.append(
                    {
                        "component_type": element_type,
                        "component_id": f"{element_type}:{element_index}",
                        "risk_channel": "overload",
                        "severity": excess,
                        "observed_value": float(value),
                    }
                )

    under_total = 0.0
    over_total = 0.0
    min_voltage = float("nan")
    max_voltage = float("nan")
    if config.pf_mode == "ac" and hasattr(net, "res_bus") and not net.res_bus.empty:
        voltage = pd.to_numeric(net.res_bus.get("vm_pu"), errors="coerce")
        finite_voltage = voltage[np.isfinite(voltage)]
        if not finite_voltage.empty:
            min_voltage = float(finite_voltage.min())
            max_voltage = float(finite_voltage.max())
        for bus, value in voltage.items():
            if not np.isfinite(value):
                continue
            under = max(config.v_min_pu - float(value), 0.0)
            over = max(float(value) - config.v_max_pu, 0.0)
            under_total += under
            over_total += over
            if under > 0.0:
                component_rows.append(
                    {
                        "component_type": "bus",
                        "component_id": f"bus:{bus}",
                        "risk_channel": "undervoltage",
                        "severity": under,
                        "observed_value": float(value),
                    }
                )
            if over > 0.0:
                component_rows.append(
                    {
                        "component_type": "bus",
                        "component_id": f"bus:{bus}",
                        "risk_channel": "overvoltage",
                        "severity": over,
                        "observed_value": float(value),
                    }
                )

    max_overload_excess = max(max_loading - config.loading_limit_percent, 0.0)
    max_under_excess = max(config.v_min_pu - min_voltage, 0.0) if np.isfinite(min_voltage) else 0.0
    max_over_excess = max(max_voltage - config.v_max_pu, 0.0) if np.isfinite(max_voltage) else 0.0
    worst_score = max(
        max_overload_excess,
        config.voltage_weight * max_under_excess,
        config.voltage_weight * max_over_excess,
    )
    composite = overload_total + config.voltage_weight * (under_total + over_total)
    return {
        "overload_excess_percent": overload_total,
        "undervoltage_excess_pu": under_total,
        "overvoltage_excess_pu": over_total,
        "nonconvergence": 0.0,
        "max_loading_percent": max_loading,
        "min_voltage_pu": min_voltage,
        "max_voltage_pu": max_voltage,
        "worst_performance_score": worst_score,
        "composite_severity": composite,
    }, component_rows


def _normalise_profiles(net: Any, baseline_operating_states: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(baseline_operating_states).copy()
    if frame.shape[1] != len(net.load):
        raise ValueError(
            "baseline_operating_states must contain one column per pandapower load "
            f"({frame.shape[1]} supplied, {len(net.load)} required)."
        )
    frame.columns = net.load.index
    frame = frame.apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Baseline load profiles must be finite and non-negative.")
    return frame


def run_time_series_pra(
    net: Any,
    baseline_operating_states: pd.DataFrame | np.ndarray,
    contingencies: Sequence[Any],
    *,
    config: PRAConfig | None = None,
    operating_state_sampler: OperatingStateSampler | None = None,
    probability_model: ContingencyProbabilityModel | None = None,
    weather_model: Any | None = None,
) -> TimeSeriesPRAResult:
    """Execute Algorithm 2 for all operating points and contingencies."""

    config = config or PRAConfig()
    profiles = _normalise_profiles(net, baseline_operating_states)
    contingency_set = normalize_contingencies(contingencies)
    if not contingency_set:
        raise ValueError("At least one contingency is required.")
    sampler = operating_state_sampler or GaussianRelativeLoadSampler()
    probability_model = probability_model or UniformProbabilityModel()
    rng = np.random.default_rng(config.seed)

    if weather_model is None:
        weather = np.full((config.n_samples, len(profiles)), None, dtype=object)
        weather_status = "not used"
    else:
        weather = weather_model.sample_regimes(
            config.n_samples, len(profiles), random_state=config.seed + 1
        )
        metadata = getattr(weather_model, "metadata", None)
        weather_status = str(getattr(metadata, "status", "provided"))

    nominal_p = pd.to_numeric(net.load["p_mw"], errors="coerce").to_numpy(dtype=float)
    nominal_q = pd.to_numeric(net.load.get("q_mvar", 0.0), errors="coerce")
    if np.isscalar(nominal_q):
        nominal_q = np.full(len(net.load), float(nominal_q))
    else:
        nominal_q = np.asarray(nominal_q, dtype=float)
    q_over_p = np.divide(
        nominal_q,
        nominal_p,
        out=np.zeros_like(nominal_p, dtype=float),
        where=np.abs(nominal_p) > 1e-12,
    )

    scenario_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    dispatch_methods: dict[str, int] = {}

    for time_position, (time_index, baseline_row) in enumerate(profiles.iterrows()):
        baseline = baseline_row.to_numpy(dtype=float)
        reference_net = copy.deepcopy(net)
        _set_load(reference_net, baseline, q_over_p)
        method = _run_reference_dispatch(reference_net, config)
        dispatch_methods[method] = dispatch_methods.get(method, 0) + 1
        samples = sampler.sample(
            baseline, config.n_samples, rng, time_index=time_index
        )
        if samples.shape != (config.n_samples, len(net.load)):
            raise ValueError(
                f"Sampler returned {samples.shape}; expected {(config.n_samples, len(net.load))}."
            )

        for sample_id, sampled_load in enumerate(samples):
            regime = weather[sample_id, time_position]
            probabilities = dict(
                probability_model.probabilities(
                    contingency_set,
                    operating_state_mw=sampled_load,
                    baseline_state_mw=baseline,
                    weather_regime=None if regime is None else str(regime),
                    time_index=time_index,
                    horizon_hours=config.step_hours,
                )
            )
            known_ids = {item.contingency_id for item in contingency_set}
            unknown = set(probabilities) - known_ids
            if unknown:
                raise ValueError(f"Probability model returned unknown contingencies: {sorted(unknown)}")
            probability_sum = float(sum(probabilities.values()))
            if any(value < 0 for value in probabilities.values()) or probability_sum > 1 + 1e-10:
                raise ValueError("Contingency probabilities must be non-negative and sum to at most one.")

            cases: list[tuple[str, Contingency | None, float]] = [
                ("base_case", None, max(0.0, 1.0 - probability_sum))
            ]
            cases.extend(
                (item.contingency_id, item, float(probabilities.get(item.contingency_id, 0.0)))
                for item in contingency_set
            )

            for contingency_id, contingency, probability in cases:
                probability_rows.append(
                    {
                        "time": time_index,
                        "sample_id": sample_id,
                        "weather_regime": regime,
                        "contingency_id": contingency_id,
                        "probability": probability,
                    }
                )
                case_net = copy.deepcopy(reference_net)
                _set_load(case_net, sampled_load, q_over_p)
                if contingency is not None:
                    _apply_contingency_in_place(case_net, contingency)
                error = None
                try:
                    _run_pf(case_net, config.pf_mode)
                    converged = bool(getattr(case_net, "converged", True))
                except Exception as exc:
                    converged = False
                    error = str(exc)
                severity, local_components = _extract_severity(
                    case_net, config, converged=converged
                )
                row = {
                    "time": time_index,
                    "sample_id": sample_id,
                    "weather_regime": regime,
                    "contingency_id": contingency_id,
                    "probability": probability,
                    "total_load_mw": float(np.sum(sampled_load)),
                    "converged": converged,
                    "error": error,
                    **severity,
                    "risk_overload": probability * severity["overload_excess_percent"],
                    "risk_undervoltage": probability * severity["undervoltage_excess_pu"],
                    "risk_overvoltage": probability * severity["overvoltage_excess_pu"],
                    "risk_nonconvergence": probability * severity["nonconvergence"],
                    "risk_composite": probability * severity["composite_severity"],
                }
                scenario_rows.append(row)
                for component in local_components:
                    component_rows.append(
                        {
                            "time": time_index,
                            "sample_id": sample_id,
                            "contingency_id": contingency_id,
                            "probability": probability,
                            **component,
                            "risk": probability * float(component["severity"]),
                        }
                    )

    scenarios = pd.DataFrame(scenario_rows)
    sampled_probabilities = pd.DataFrame(probability_rows)
    raw_components = pd.DataFrame(component_rows)

    sample_totals = (
        scenarios.groupby(["time", "sample_id"], as_index=False)
        .agg(
            risk_overload=("risk_overload", "sum"),
            risk_undervoltage=("risk_undervoltage", "sum"),
            risk_overvoltage=("risk_overvoltage", "sum"),
            risk_nonconvergence=("risk_nonconvergence", "sum"),
            risk_composite=("risk_composite", "sum"),
            worst_performance_score=("worst_performance_score", "max"),
            total_load_mw=("total_load_mw", "first"),
        )
    )
    risk_by_time = (
        sample_totals.groupby("time", as_index=False)
        .agg(
            risk_overload=("risk_overload", "mean"),
            risk_undervoltage=("risk_undervoltage", "mean"),
            risk_overvoltage=("risk_overvoltage", "mean"),
            risk_nonconvergence=("risk_nonconvergence", "mean"),
            risk_composite=("risk_composite", "mean"),
            mc_standard_error=("risk_composite", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
            worst_performance_score=("worst_performance_score", "max"),
            mean_total_load_mw=("total_load_mw", "mean"),
        )
    )
    risk_by_time["risk_composite_per_100mw"] = np.divide(
        100.0 * risk_by_time["risk_composite"],
        risk_by_time["mean_total_load_mw"],
        out=np.zeros(len(risk_by_time), dtype=float),
        where=risk_by_time["mean_total_load_mw"].to_numpy(dtype=float) > 0,
    )

    risk_by_contingency = (
        scenarios.groupby(["time", "contingency_id"], as_index=False)
        .agg(
            mean_probability=("probability", "mean"),
            risk_overload=("risk_overload", "mean"),
            risk_undervoltage=("risk_undervoltage", "mean"),
            risk_overvoltage=("risk_overvoltage", "mean"),
            risk_nonconvergence=("risk_nonconvergence", "mean"),
            risk_composite=("risk_composite", "mean"),
            max_severity=("composite_severity", "max"),
            failed_power_flows=("converged", lambda x: int((~x.astype(bool)).sum())),
        )
        .sort_values(["time", "risk_composite"], ascending=[True, False], ignore_index=True)
    )

    if raw_components.empty:
        risk_by_component = pd.DataFrame(
            columns=["time", "component_type", "component_id", "risk_channel", "expected_risk", "max_severity"]
        )
    else:
        risk_by_component = (
            raw_components.groupby(
                ["time", "component_type", "component_id", "risk_channel"], as_index=False
            )
            .agg(
                expected_risk=("risk", lambda x: float(np.sum(x) / config.n_samples)),
                max_severity=("severity", "max"),
            )
            .sort_values(["time", "expected_risk"], ascending=[True, False], ignore_index=True)
        )

    summary = {
        "n_operating_points": int(len(profiles)),
        "n_samples_per_operating_point": int(config.n_samples),
        "n_contingencies": int(len(contingency_set)),
        "pf_mode": config.pf_mode,
        "expected_composite_risk_sum": float(risk_by_time["risk_composite"].sum()),
        "expected_overload_risk_sum": float(risk_by_time["risk_overload"].sum()),
        "expected_undervoltage_risk_sum": float(risk_by_time["risk_undervoltage"].sum()),
        "expected_overvoltage_risk_sum": float(risk_by_time["risk_overvoltage"].sum()),
        "maximum_worst_performance_score": float(risk_by_time["worst_performance_score"].max()),
        "failed_power_flow_count": int((~scenarios["converged"].astype(bool)).sum()),
        "weather_model_status": weather_status,
        "reference_dispatch_methods": dispatch_methods,
        "config": asdict(config),
        "interpretation": (
            "Composite risk is a ranking score: overload excess is measured in percentage points, "
            "and voltage deviations are multiplied by voltage_weight. Physical risk channels are "
            "reported separately and should be used for engineering interpretation."
        ),
    }
    return TimeSeriesPRAResult(
        summary=summary,
        risk_by_time=risk_by_time,
        risk_by_contingency=risk_by_contingency,
        risk_by_component=risk_by_component,
        scenario_results=scenarios,
        sampled_probabilities=sampled_probabilities,
    )
