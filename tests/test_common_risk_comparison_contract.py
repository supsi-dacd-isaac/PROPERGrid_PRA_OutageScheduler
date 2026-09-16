from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gurobi]
pytest.importorskip("gurobipy")

from scheduler_clustered_v0.clustered_cvar_engine import ClusteredCVaRConfig  # noqa: E402
from scheduler_clustered_v0.common_risk_comparison import (  # noqa: E402
    CommonPoolCandidateRecord,
    CommonRiskComparisonConfig,
    CommonRiskComparisonResult,
    SelectedCommonPoolSchedule,
)
from scheduler_clustered_v0.common_risk_selection import PairedLossSummary  # noqa: E402
from scheduler_clustered_v0.risk_metrics import RiskSummary  # noqa: E402


def make_risk(expected: float, cvar: float) -> RiskSummary:
    return RiskSummary(
        expected_loss=expected,
        var=cvar,
        cvar=cvar,
        alpha=0.95,
        tail_ids=("s1",),
        normalized_probabilities={"s1": 1.0},
    )


def make_selected(label: str, index: int, start: str, expected: float, cvar: float):
    risk = make_risk(expected, cvar)
    candidate = CommonPoolCandidateRecord(
        pool_index=index,
        master_objective=10.0,
        selection_utility=10.0,
        maintenance_utility=10.0,
        start_times={"line_0": start},
        deferred_outages=(),
        active_outages={},
        risk=risk,
        scenario_losses={"s1": cvar},
        clusters=[],
    )
    return SelectedCommonPoolSchedule(
        label=label,
        selection_metric=label,
        candidate=candidate,
        validation_risk=risk,
        validation_losses={"s1": cvar},
        validation_clusters=[],
    )


def test_result_contract_reports_cvar_winner_and_strict_json() -> None:
    risk_neutral = make_selected("risk_neutral", 1, "t0", 5.0, 10.0)
    cvar = make_selected("cvar", 2, "t1", 6.0, 8.0)
    paired = PairedLossSummary(
        expected_difference=-2.0,
        minimum_difference=-2.0,
        maximum_difference=-2.0,
        probability_cvar_better=1.0,
        probability_risk_neutral_better=0.0,
        probability_tie=0.0,
        scenario_differences={"s1": -2.0},
    )
    result = CommonRiskComparisonResult(
        risk_neutral=risk_neutral,
        cvar=cvar,
        candidate_pool=[risk_neutral.candidate, cvar.candidate],
        eligible_candidate_indices=(1, 2),
        utility_floor=9.9,
        scenario_probabilities={"s1": 1.0},
        paired_validation=paired,
        termination_reason="candidate_pool_limit",
        configuration=CommonRiskComparisonConfig(),
        risk_configuration=ClusteredCVaRConfig(),
    )
    payload = result.to_dict()
    assert result.same_schedule is False
    assert result.validation_cvar_difference == pytest.approx(-2.0)
    assert payload["tail_risk_winner"] == "cvar"
    assert payload["validation_differences_cvar_minus_risk_neutral"]["cvar_mw"] == pytest.approx(-2.0)
    json.dumps(payload, allow_nan=False)
