from __future__ import annotations

import pandas as pd
import pytest

from pra_psa.severity import (
    branch_overload_excess,
    compute_severity,
    max_branch_loading,
    number_of_overloads,
    summarize_branch_state,
    voltage_violation_excess,
)


@pytest.fixture
def branch_results() -> pd.DataFrame:
    return pd.DataFrame({"loading_percent": [80.0, 101.0, 125.0, float("nan")]})


@pytest.fixture
def bus_results() -> pd.DataFrame:
    return pd.DataFrame({"vm_pu": [0.90, 0.98, 1.10, float("nan")]})


@pytest.mark.unit
def test_branch_severity_metrics(branch_results: pd.DataFrame) -> None:
    assert branch_overload_excess(branch_results) == pytest.approx(26.0)
    assert max_branch_loading(branch_results) == pytest.approx(125.0)
    assert number_of_overloads(branch_results) == 2
    assert summarize_branch_state(branch_results) == {
        "max_loading_pct": pytest.approx(125.0),
        "n_overloads": 2,
        "overload_excess": pytest.approx(26.0),
    }


@pytest.mark.unit
def test_voltage_and_combined_severity(
    branch_results: pd.DataFrame,
    bus_results: pd.DataFrame,
) -> None:
    voltage = voltage_violation_excess(bus_results, low_vm_pu=0.92, high_vm_pu=1.08)
    assert voltage == pytest.approx(0.04)
    combined = compute_severity(
        branch_results,
        bus_results,
        metric="combined_overload_voltage",
    )
    assert combined == pytest.approx(30.0)


@pytest.mark.unit
def test_ens_and_empty_inputs() -> None:
    assert compute_severity(pd.DataFrame(), metric="ens_mwh", dns_mw=5.0, duration_h=3.0) == 15.0
    assert branch_overload_excess(pd.DataFrame()) == 0.0
    assert number_of_overloads(pd.DataFrame()) == 0


@pytest.mark.unit
def test_unknown_severity_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported severity metric"):
        compute_severity(pd.DataFrame(), metric="unknown")
