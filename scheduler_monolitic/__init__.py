"""Risk-informed monolithic outage scheduling for transmission systems.

The package keeps its numerical risk utilities importable without Gurobi. Solver
objects are loaded lazily through :func:`solve_scos`.
"""

from __future__ import annotations

from .config import ModelConfig, OutputConfig, RiskConfig, ScenarioConfig, SolverConfig
from .risk_measures import RiskSummary, summarise_losses, weighted_cvar, weighted_quantile
from .scenario_generation import ScenarioBank, generate_demand_scenarios
from .model_size import ModelSizeEstimate, estimate_extensive_form_size

__all__ = [
    "ModelConfig",
    "OutputConfig",
    "RiskConfig",
    "ScenarioBank",
    "ScenarioConfig",
    "SolverConfig",
    "RiskSummary",
    "generate_demand_scenarios",
    "ModelSizeEstimate",
    "estimate_extensive_form_size",
    "solve_scos",
    "summarise_losses",
    "weighted_cvar",
    "weighted_quantile",
]


def solve_scos(*args, **kwargs):
    """Lazily import and call the Gurobi-backed solver."""
    from .monolithic_scos import solve_scos as _solve_scos

    return _solve_scos(*args, **kwargs)
