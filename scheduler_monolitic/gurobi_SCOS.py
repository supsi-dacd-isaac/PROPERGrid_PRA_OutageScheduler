"""Backward-compatible entry point for the risk-neutral monolithic scheduler."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .config import ModelConfig, RiskConfig, ScenarioConfig, SolverConfig
from .monolithic_scos import solve_scos
from .scenario_generation import ScenarioBank, generate_demand_scenarios


def deterministic_SCOS_gurobi(
    data: dict[str, Any],
    VOLL: float = 1.0e7,
    use_DC_PF: bool = True,
    save_res_name: str | Path | None = None,
    *,
    scenario_bank: ScenarioBank | None = None,
    n_samples: int | None = None,
    seed: int = 42,
    contingency_top_k: int | None = None,
):
    """
    Solve the common stochastic model using expected schedule loss.

    ``VOLL`` is retained for API compatibility but objective scaling is now
    dimensionless and explicit. Pass the same ``scenario_bank`` to the CVaR
    wrapper for a controlled comparison.
    """
    warnings.warn(
        "scheduler_monolitic.gurobi_SCOS is a compatibility wrapper; use scheduler_monolitic.monolithic_scos instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    del VOLL
    if scenario_bank is None:
        count = int(n_samples or data.get("n_samples", data.get("config", {}).get("n_samples", 20)))
        scenario_bank = generate_demand_scenarios(
            data["nodal_demand"],
            ScenarioConfig(n_scenarios=count, seed=seed),
        )
    result = solve_scos(
        data,
        scenario_bank,
        risk_config=RiskConfig(formulation="expected"),
        model_config=ModelConfig(
            use_dc_power_flow=use_DC_PF,
            contingency_top_k=contingency_top_k,
            store_full_solution=True,
        ),
        solver_config=SolverConfig(seed=seed),
    )
    if save_res_name is not None and result.raw_solution is not None:
        import json

        path = Path(save_res_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.raw_solution), encoding="utf-8")
    return result.as_legacy_dictionary(), result.raw_solution
