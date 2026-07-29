from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pra_psa.core.models import Contingency, Outage
from pra_psa.risk import (
    HomogeneousPoissonContingencyModel,
    UniformContingencyModel,
    aggregate_risk,
    build_risk_curve,
)


@pytest.mark.unit
def test_uniform_contingency_probability_mass() -> None:
    contingencies = [Contingency((Outage("line", i),)) for i in range(4)]
    probabilities = UniformContingencyModel(0.08).get_probabilities(contingencies)
    assert set(probabilities) == {f"line:{i}" for i in range(4)}
    assert sum(probabilities.values()) == pytest.approx(0.08)
    assert all(value == pytest.approx(0.02) for value in probabilities.values())


@pytest.mark.unit
def test_poisson_model_uses_component_rates_and_caps_total_mass() -> None:
    contingencies = [
        Contingency((Outage("line", 0),), contingency_id="l0"),
        Contingency((Outage("trafo", 1),), contingency_id="t1"),
    ]
    model = HomogeneousPoissonContingencyModel(
        {"line": 0.1, "trafo": 0.2},
        exposure_time=2.0,
        max_total_contingency_probability=0.10,
    )
    probabilities = model.get_probabilities(contingencies)
    assert sum(probabilities.values()) == pytest.approx(0.10)
    assert probabilities["t1"] > probabilities["l0"]


@pytest.mark.unit
def test_risk_aggregation_assigns_remaining_mass_to_base_case() -> None:
    summary = pd.DataFrame(
        [
            {"time": "t0", "contingency_id": "base", "severity": 2.0, "converged": True, "n_overloads": 0},
            {"time": "t0", "contingency_id": "c1", "severity": 10.0, "converged": True, "n_overloads": 1},
            {"time": "t1", "contingency_id": "base", "severity": 1.0, "converged": True, "n_overloads": 0},
            {"time": "t1", "contingency_id": "c1", "severity": 20.0, "converged": False, "n_overloads": 2},
        ]
    )
    result = aggregate_risk(
        summary,
        probabilities={"c1": 0.1},
        base_case_id="base",
        alpha=0.75,
    )
    by_id = result.risk_by_contingency.set_index("contingency_id")
    assert by_id.at["base", "probability"] == pytest.approx(0.9)
    assert by_id.at["c1", "probability"] == pytest.approx(0.1)
    assert by_id.at["c1", "n_failed_pf"] == 1

    by_time = result.risk_by_time.set_index("time")
    assert by_time.at["t0", "total_risk"] == pytest.approx(2.8)
    assert by_time.at["t1", "total_risk"] == pytest.approx(2.9)


@pytest.mark.unit
def test_risk_curve_is_sorted_and_reports_common_var_cvar() -> None:
    severity = np.asarray([10.0, 0.0, 100.0, 20.0])
    curve = build_risk_curve(severity, alpha=0.75)
    assert curve["severity"].tolist() == [0.0, 10.0, 20.0, 100.0]
    assert curve["exceedance_probability"].is_monotonic_decreasing
    assert curve["VaR"].nunique() == 1
    assert curve["CVaR"].nunique() == 1
    assert curve["VaR"].iloc[0] == pytest.approx(np.quantile(severity, 0.75))
    assert curve["CVaR"].iloc[0] == pytest.approx(100.0)


@pytest.mark.unit
def test_contingency_models_validate_probability_bounds() -> None:
    with pytest.raises(ValueError):
        UniformContingencyModel(1.1)
