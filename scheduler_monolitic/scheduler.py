"""Compatibility class around the unified risk-neutral scheduler."""

from __future__ import annotations

from typing import Any

from .config import ModelConfig, RiskConfig, ScenarioConfig, SolverConfig
from .monolithic_scos import solve_scos
from .scenario_generation import generate_demand_scenarios


class outage_scheduler:  # noqa: N801 - retained for backward compatibility
    """Legacy class name retained while delegating to the unified model."""

    def __init__(self, data: dict[str, Any]):
        self.data = dict(data)
        self.result = None

    def build_model(self, nodal_demand):
        self.data["nodal_demand"] = nodal_demand
        return self

    def solve(self, save_res_name=None):
        count = int(self.data.get("config", {}).get("n_samples", 20))
        bank = generate_demand_scenarios(
            self.data["nodal_demand"],
            ScenarioConfig(n_scenarios=count, seed=42),
        )
        self.result = solve_scos(
            self.data,
            bank,
            risk_config=RiskConfig(formulation="expected"),
            model_config=ModelConfig(),
            solver_config=SolverConfig(),
        )
        return self.result.as_legacy_dictionary(), self.result.raw_solution
