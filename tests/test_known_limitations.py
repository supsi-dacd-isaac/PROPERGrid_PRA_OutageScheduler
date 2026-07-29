"""Executable documentation of intentionally incomplete legacy APIs.

These xfails prevent placeholder modules from being mistaken for tested,
production-ready functionality while keeping the default suite green.
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="subset simulation is currently a placeholder")
def test_subset_simulation_placeholder_is_not_a_valid_result() -> None:
    pytest.importorskip("pandapower")
    from pra_psa.sampler.subset_simulation import SubsetSym

    assert SubsetSym(object(), [], lambda _: 0.0) is not None


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="unified Scheduler.optimize remains a placeholder")
def test_unified_scheduler_optimize_is_implemented() -> None:
    pytest.importorskip("pandapower")
    from pra_psa import Scheduler

    scheduler = Scheduler(grid_data=object())
    scheduler.optimize()
