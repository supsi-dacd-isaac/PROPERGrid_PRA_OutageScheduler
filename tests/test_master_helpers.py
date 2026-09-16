from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gurobi]
pytest.importorskip("gurobipy")

from scheduler_clustered_v0.master_problem import (  # noqa: E402
    MasterSolution,
    covering_starts,
    duration_steps,
    feasible_starts,
    no_good_from_solution,
)


def base_data() -> dict:
    return {
        "T": [f"t{i}" for i in range(6)],
        "names": {"outages": ["A", "B"]},
        "durations": {"A": 2.4, "B": 1.0},
        "outage_start_windows": {
            "A": {"start": "t1", "end": "t4", "end_is_completion": True},
            "B": [0, 5],
        },
    }


def test_duration_rounding_and_validation() -> None:
    data = base_data()
    assert duration_steps(data, "A") == 2
    data["durations"]["A"] = 0
    with pytest.raises(ValueError, match="non-positive"):
        duration_steps(data, "A")


def test_feasible_start_windows_and_covering_starts() -> None:
    data = base_data()
    starts = feasible_starts(data)
    assert starts["A"] == ("t1", "t2", "t3")
    assert starts["B"] == ("t0", "t1", "t2", "t3", "t4", "t5")

    cover = covering_starts(data, starts)
    assert cover[("t1", "A")] == ("t1",)
    assert cover[("t2", "A")] == ("t1", "t2")
    assert cover[("t4", "A")] == ("t3",)


def test_no_good_cut_preserves_decision_signature() -> None:
    solution = MasterSolution(
        status=2,
        objective=1.0,
        start_times={"B": "t2", "A": "t1"},
        deferred_outages=("C",),
        active_outages={},
        maintenance_utility=1.0,
        proxy_security_penalty=0.0,
    )
    cut = no_good_from_solution(solution, label="test")
    assert cut.starts == (("A", "t1"), ("B", "t2"))
    assert cut.deferred == ("C",)
    assert cut.label == "test"
