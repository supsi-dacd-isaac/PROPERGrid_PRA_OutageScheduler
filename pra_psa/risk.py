"""Risk aggregation models and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import numpy as np
import pandas as pd

from pra_psa.core.models import normalize_contingencies


@dataclass
class RiskAggregationResult:
    risk_by_contingency: pd.DataFrame
    risk_by_time: pd.DataFrame
    risk_curve: pd.DataFrame

    def to_csv(self, folder: str) -> None:
        from pathlib import Path

        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        self.risk_by_contingency.to_csv(folder_path / "risk_by_contingency.csv", index=False)
        self.risk_by_time.to_csv(folder_path / "risk_by_time.csv", index=False)
        self.risk_curve.to_csv(folder_path / "risk_curve.csv", index=False)


class UniformContingencyModel:
    """Assign equal total probability mass to all supplied contingencies."""

    def __init__(self, total_contingency_probability: float = 0.01):
        if not (0 <= total_contingency_probability <= 1):
            raise ValueError("total_contingency_probability must be in [0, 1]")
        self.total_contingency_probability = float(total_contingency_probability)

    def get_probabilities(self, contingencies: Iterable[Any]) -> dict[str, float]:
        cont_list = normalize_contingencies(contingencies)
        if not cont_list:
            return {}
        p = self.total_contingency_probability / len(cont_list)
        return {c.contingency_id: p for c in cont_list}


class HomogeneousPoissonContingencyModel:
    """Simple homogeneous Poisson occurrence model.

    Failure rates can be supplied per outage ID (``"line:3"``), per element
    type (``"line"``), or as a scalar default.
    """

    def __init__(
        self,
        failure_rates: Mapping[str, float] | float = 1e-4,
        *,
        exposure_time: float = 1.0,
        max_total_contingency_probability: float = 0.2,
    ):
        self.failure_rates = failure_rates
        self.exposure_time = float(exposure_time)
        self.max_total_contingency_probability = float(max_total_contingency_probability)

    def get_probabilities(self, contingencies: Iterable[Any]) -> dict[str, float]:
        cont_list = normalize_contingencies(contingencies)
        probs = {}
        for c in cont_list:
            p = 1.0
            for outage in c.outages:
                lam = self._rate(outage.short_id, outage.element_type)
                p *= 1.0 - np.exp(-lam * self.exposure_time)
            probs[c.contingency_id] = float(p)
        total = sum(probs.values())
        if total > self.max_total_contingency_probability and total > 0:
            scale = self.max_total_contingency_probability / total
            probs = {k: v * scale for k, v in probs.items()}
        return probs

    def _rate(self, outage_id: str, element_type: str) -> float:
        if isinstance(self.failure_rates, Mapping):
            if outage_id in self.failure_rates:
                return float(self.failure_rates[outage_id])
            if element_type in self.failure_rates:
                return float(self.failure_rates[element_type])
            return float(self.failure_rates.get("default", 1e-4))
        return float(self.failure_rates)


def aggregate_risk(
    contingency_summary: pd.DataFrame,
    probabilities: Mapping[str, float] | None = None,
    *,
    severity_col: str = "severity",
    time_col: str = "time",
    contingency_col: str = "contingency_id",
    base_case_id: str = "base",
    alpha: float = 0.95,
) -> RiskAggregationResult:
    """Aggregate contingency-analysis schedule_results into PRA risk tables."""
    if contingency_summary.empty:
        empty = pd.DataFrame()
        return RiskAggregationResult(empty, empty, empty)
    df = contingency_summary.copy()
    if probabilities is None:
        probabilities = {}
    df["probability"] = df[contingency_col].map(probabilities).fillna(0.0)
    if base_case_id in set(df[contingency_col]):
        # The normal-state probability is the remaining probability mass.
        normal_probability = max(0.0, 1.0 - sum(v for k, v in probabilities.items() if k != base_case_id))
        df.loc[df[contingency_col] == base_case_id, "probability"] = normal_probability
    df["risk"] = df["probability"] * df[severity_col].astype(float)

    risk_by_contingency = (
        df.groupby(contingency_col, dropna=False)
        .agg(
            probability=("probability", "mean"),
            mean_severity=(severity_col, "mean"),
            max_severity=(severity_col, "max"),
            mean_risk=("risk", "mean"),
            max_risk=("risk", "max"),
            n_cases=(severity_col, "size"),
            n_failed_pf=("converged", lambda s: int((~s.astype(bool)).sum()) if s.notna().any() else 0),
        )
        .reset_index()
        .sort_values("mean_risk", ascending=False)
    )

    risk_by_time = (
        df.groupby(time_col, dropna=False)
        .agg(
            total_risk=("risk", "sum"),
            mean_severity=(severity_col, "mean"),
            max_severity=(severity_col, "max"),
            n_unsafe=("n_overloads", lambda s: int(np.nansum(np.asarray(s, dtype=float) > 0))),
        )
        .reset_index()
    )

    risk_curve = build_risk_curve(df[severity_col].astype(float).to_numpy(), alpha=alpha)
    return RiskAggregationResult(risk_by_contingency, risk_by_time, risk_curve)


def build_risk_curve(severity: np.ndarray, *, alpha: float = 0.95) -> pd.DataFrame:
    severity = np.asarray(severity, dtype=float)
    severity = severity[np.isfinite(severity)]
    if severity.size == 0:
        return pd.DataFrame(columns=["severity", "exceedance_probability", "VaR", "CVaR"])
    sorted_sev = np.sort(severity)
    n = len(sorted_sev)
    exceedance = 1.0 - (np.arange(1, n + 1) / n)
    var = float(np.quantile(sorted_sev, alpha))
    tail = sorted_sev[sorted_sev >= var]
    cvar = float(tail.mean()) if tail.size else var
    out = pd.DataFrame({"severity": sorted_sev, "exceedance_probability": exceedance})
    out["VaR"] = var
    out["CVaR"] = cvar
    return out
