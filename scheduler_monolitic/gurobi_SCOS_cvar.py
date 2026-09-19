"""Backward-compatible entry point for the corrected CVaR monolithic scheduler."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .config import ModelConfig, RiskConfig, ScenarioConfig, SolverConfig
from .monolithic_scos import solve_scos
from .scenario_generation import ScenarioBank, generate_demand_scenarios


def CVAR_SCOS_gurobi(
    data: dict[str, Any],
    n_samples: int = 20,
    alpha: float = 0.95,
    VOLL: float = 1.0e7,
    use_DC_PF: bool = True,
    save_res_name: str | Path | None = None,
    *,
    scenario_bank: ScenarioBank | None = None,
    seed: int = 42,
    contingency_top_k: int | None = None,
    reference_schedule=None,
    reference_utility: float | None = None,
):
    """
    Solve the corrected system-level CVaR formulation.

    ``alpha`` is interpreted as the CVaR confidence level, e.g. 0.95. The old
    implementation's ``1/(n*alpha)`` convention is no longer supported.
    """
    warnings.warn(
        "scheduler_monolitic.gurobi_SCOS_cvar is a compatibility wrapper; use scheduler_monolitic.monolithic_scos instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    del VOLL
    if scenario_bank is None:
        scenario_bank = generate_demand_scenarios(
            data["nodal_demand"],
            ScenarioConfig(n_scenarios=n_samples, seed=seed),
        )
    result = solve_scos(
        data,
        scenario_bank,
        risk_config=RiskConfig(beta=alpha, formulation="cvar"),
        model_config=ModelConfig(
            use_dc_power_flow=use_DC_PF,
            contingency_top_k=contingency_top_k,
            store_full_solution=True,
        ),
        solver_config=SolverConfig(seed=seed),
        reference_schedule=reference_schedule,
        reference_utility=reference_utility,
    )
    if save_res_name is not None and result.raw_solution is not None:
        import json

        path = Path(save_res_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.raw_solution), encoding="utf-8")
    return result.as_legacy_dictionary(), result.raw_solution


def demand_sampler(nodal_demand, n_samples: int, random_seed: int = 42):
    """Compatibility adapter returning a list of sampled demand DataFrames."""
    bank = generate_demand_scenarios(
        nodal_demand,
        ScenarioConfig(n_scenarios=n_samples, seed=random_seed),
    )
    return [bank.demands[label] for label in bank.labels]
