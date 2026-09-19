"""Monte Carlo probabilistic risk-assessment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from probabilistic_models.base import NodalScenarioModel, as_2d_float
from probabilistic_models.contingencies.contingencies import PoissonFailureModel
from probabilistic_models.weather.weather import PlaceholderWeatherModel, WeatherRegimeModel

from .cascade import CascadingFailureSimulator


@dataclass(frozen=True)
class Contingency:
    contingency_id: str
    asset_ids: tuple[str, ...]
    direct_rate_per_hour: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.asset_ids:
            raise ValueError("A contingency must contain at least one asset.")
        if self.direct_rate_per_hour is not None and self.direct_rate_per_hour < 0:
            raise ValueError("direct_rate_per_hour cannot be negative.")


@dataclass(frozen=True)
class PRASummary:
    n_operational_scenarios: int
    n_contingencies: int
    horizon_hours: float
    expected_dns_mwh: float
    probability_of_positive_dns: float
    maximum_dns_mw: float
    monte_carlo_standard_error_mwh: float
    monte_carlo_ci95_lower_mwh: float
    monte_carlo_ci95_upper_mwh: float
    mean_enumerated_probability_mass: float
    mean_omitted_probability_mass: float | None
    weather_model_status: str
    rate_assumptions: tuple[str, ...]


@dataclass
class PRAResult:
    summary: PRASummary
    risk_by_contingency: pd.DataFrame
    exceedance_curve: pd.DataFrame
    component_importance: pd.DataFrame
    scenario_results: pd.DataFrame

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "pra_summary.json").write_text(
            json.dumps(asdict(self.summary), indent=2), encoding="utf-8"
        )
        self.risk_by_contingency.to_csv(output / "risk_by_contingency.csv", index=False)
        self.exceedance_curve.to_csv(output / "risk_exceedance_curve.csv", index=False)
        self.component_importance.to_csv(output / "component_importance.csv", index=False)
        self.scenario_results.to_csv(output / "pra_scenario_results.csv", index=False)


def build_n1_branch_contingencies(simulator: CascadingFailureSimulator) -> list[Contingency]:
    return [
        Contingency(f"N-1:{asset_id}", (asset_id,), description="single branch outage")
        for asset_id in simulator.network.branch_ids
    ]


class PRAEngine:
    """Combine nodal scenarios, event probabilities, and cascade consequences."""

    def __init__(
        self,
        simulator: CascadingFailureSimulator,
        failure_model: PoissonFailureModel,
        *,
        nodal_model: NodalScenarioModel | None = None,
        weather_model: WeatherRegimeModel | None = None,
        contingencies: list[Contingency] | None = None,
    ):
        self.simulator = simulator
        self.failure_model = failure_model
        self.nodal_model = nodal_model
        self.weather_model = weather_model or PlaceholderWeatherModel()
        self.contingencies = contingencies or build_n1_branch_contingencies(simulator)
        known = set(failure_model.by_id)
        for contingency in self.contingencies:
            if contingency.direct_rate_per_hour is None:
                unknown = set(contingency.asset_ids) - known
                if unknown:
                    raise KeyError(
                        f"Contingency {contingency.contingency_id!r} contains assets absent from the failure model: "
                        f"{sorted(unknown)}"
                    )
        subsets = [tuple(sorted(item.asset_ids)) for item in self.contingencies if item.direct_rate_per_hour is None]
        if len(subsets) != len(set(subsets)):
            raise ValueError("Independent-Poisson contingencies must have unique asset subsets.")

    def _load_samples(
        self,
        n_samples: int,
        load_samples,
        random_state: int,
    ) -> np.ndarray:
        if load_samples is not None:
            samples = as_2d_float(load_samples, name="load_samples")
            if samples.shape[0] != n_samples:
                raise ValueError("load_samples row count must equal n_samples.")
        elif self.nodal_model is not None:
            samples = self.nodal_model.sample(n_samples, random_state=random_state)
        else:
            samples = np.repeat(self.simulator.network.load_mw[None, :], n_samples, axis=0)
        if samples.shape[1] != self.simulator.network.n_buses:
            raise ValueError(
                f"Nodal samples contain {samples.shape[1]} buses; network has {self.simulator.network.n_buses}."
            )
        return samples

    def _probability(self, contingency: Contingency, horizon: float, regime: str) -> float:
        if contingency.direct_rate_per_hour is not None:
            return float(1.0 - np.exp(-contingency.direct_rate_per_hour * horizon))
        return self.failure_model.contingency_probability(
            contingency.asset_ids, horizon, regime, exact_subset=True
        )

    @staticmethod
    def _exceedance_curve(records: pd.DataFrame) -> pd.DataFrame:
        severities = records["dns_mw"].to_numpy(dtype=float)
        weights = records["probability_weight"].to_numpy(dtype=float) / records["sample_count"].to_numpy(dtype=float)
        thresholds = np.unique(np.concatenate([[0.0], severities]))
        exceedance = np.array([weights[severities > threshold].sum() for threshold in thresholds])
        return pd.DataFrame(
            {"dns_threshold_mw": thresholds, "exceedance_probability": exceedance}
        ).sort_values("dns_threshold_mw", ignore_index=True)

    def run(
        self,
        *,
        n_samples: int = 500,
        horizon_hours: float = 1.0,
        load_samples=None,
        planned_outages: Iterable[str] = (),
        include_normal_state: bool = True,
        random_state: int = 42,
    ) -> PRAResult:
        if n_samples < 2 or horizon_hours <= 0:
            raise ValueError("n_samples must be at least 2 and horizon_hours must be positive.")
        planned = tuple(dict.fromkeys(map(str, planned_outages)))
        loads = self._load_samples(int(n_samples), load_samples, random_state)
        regimes = self.weather_model.sample_regimes(
            int(n_samples), 1, random_state=random_state + 1
        ).ravel()
        records: list[dict[str, object]] = []
        total_risk_by_sample = np.zeros(int(n_samples))
        positive_dns_probability_by_sample = np.zeros(int(n_samples))
        enumerated_mass = np.zeros(int(n_samples))
        omitted_mass_available = not any(
            contingency.direct_rate_per_hour is not None for contingency in self.contingencies
        )

        for sample_id, (load, regime) in enumerate(zip(loads, regimes)):
            if include_normal_state:
                probability = self.failure_model.normal_probability(horizon_hours, str(regime))
                result = self.simulator.simulate(
                    load_mw=load, initiating_outages=(), planned_outages=planned
                )
                dns = result.demand_not_served_mw
                risk = probability * dns * horizon_hours
                records.append(
                    {
                        "sample_id": sample_id,
                        "sample_count": int(n_samples),
                        "weather_regime": str(regime),
                        "contingency_id": "normal_forced_state",
                        "initiating_assets": "",
                        "probability_weight": probability,
                        "dns_mw": dns,
                        "risk_dns_mwh": risk,
                        "propagated_assets": ";".join(result.propagated_outages),
                        "cascade_stages": len(result.stages) - 1,
                    }
                )
                total_risk_by_sample[sample_id] += risk
                positive_dns_probability_by_sample[sample_id] += probability * float(dns > 0)
                enumerated_mass[sample_id] += probability

            for contingency in self.contingencies:
                probability = self._probability(contingency, horizon_hours, str(regime))
                result = self.simulator.simulate(
                    load_mw=load,
                    initiating_outages=contingency.asset_ids,
                    planned_outages=planned,
                )
                dns = result.demand_not_served_mw
                risk = probability * dns * horizon_hours
                records.append(
                    {
                        "sample_id": sample_id,
                        "sample_count": int(n_samples),
                        "weather_regime": str(regime),
                        "contingency_id": contingency.contingency_id,
                        "initiating_assets": ";".join(contingency.asset_ids),
                        "probability_weight": probability,
                        "dns_mw": dns,
                        "risk_dns_mwh": risk,
                        "propagated_assets": ";".join(result.propagated_outages),
                        "cascade_stages": len(result.stages) - 1,
                    }
                )
                total_risk_by_sample[sample_id] += risk
                positive_dns_probability_by_sample[sample_id] += probability * float(dns > 0)
                enumerated_mass[sample_id] += probability

        scenario_results = pd.DataFrame(records)
        risk_by_contingency = (
            scenario_results.groupby("contingency_id", sort=False)
            .agg(
                mean_probability=("probability_weight", "mean"),
                mean_dns_mw=("dns_mw", "mean"),
                maximum_dns_mw=("dns_mw", "max"),
                expected_dns_mwh=("risk_dns_mwh", "mean"),
                mean_cascade_stages=("cascade_stages", "mean"),
            )
            .reset_index()
            .sort_values("expected_dns_mwh", ascending=False, ignore_index=True)
        )
        exceedance = self._exceedance_curve(scenario_results)

        importance: dict[str, dict[str, float]] = {
            asset_id: {"initiating_risk_mwh": 0.0, "weighted_propagation_count": 0.0}
            for asset_id in self.failure_model.by_id
        }
        for row in scenario_results.itertuples(index=False):
            for asset_id in filter(None, str(row.initiating_assets).split(";")):
                importance.setdefault(asset_id, {"initiating_risk_mwh": 0.0, "weighted_propagation_count": 0.0})
                importance[asset_id]["initiating_risk_mwh"] += float(row.risk_dns_mwh) / n_samples
            for asset_id in filter(None, str(row.propagated_assets).split(";")):
                importance.setdefault(asset_id, {"initiating_risk_mwh": 0.0, "weighted_propagation_count": 0.0})
                importance[asset_id]["weighted_propagation_count"] += float(row.probability_weight) / n_samples
        component_importance = pd.DataFrame(
            [{"asset_id": asset_id, **values} for asset_id, values in importance.items()]
        ).sort_values(
            ["initiating_risk_mwh", "weighted_propagation_count"],
            ascending=False,
            ignore_index=True,
        )

        expected_risk = float(total_risk_by_sample.mean())
        standard_error = float(total_risk_by_sample.std(ddof=1) / np.sqrt(n_samples))
        omitted_mass = (
            float(np.mean(np.maximum(1.0 - enumerated_mass, 0.0)))
            if omitted_mass_available
            else None
        )
        summary = PRASummary(
            n_operational_scenarios=int(n_samples),
            n_contingencies=len(self.contingencies),
            horizon_hours=float(horizon_hours),
            expected_dns_mwh=expected_risk,
            probability_of_positive_dns=float(positive_dns_probability_by_sample.mean()),
            maximum_dns_mw=float(scenario_results["dns_mw"].max()),
            monte_carlo_standard_error_mwh=standard_error,
            monte_carlo_ci95_lower_mwh=max(expected_risk - 1.96 * standard_error, 0.0),
            monte_carlo_ci95_upper_mwh=expected_risk + 1.96 * standard_error,
            mean_enumerated_probability_mass=float(enumerated_mass.mean()),
            mean_omitted_probability_mass=omitted_mass,
            weather_model_status=self.weather_model.metadata.status,
            rate_assumptions=tuple(self.failure_model.assumption_warnings),
        )
        return PRAResult(summary, risk_by_contingency, exceedance, component_importance, scenario_results)

