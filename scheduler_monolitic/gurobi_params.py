"""Backward-compatible access to centralised Gurobi parameters."""

from __future__ import annotations

from .config import SolverConfig


def _as_gurobi_dictionary(config: SolverConfig) -> dict[str, object]:
    return {
        "MIPGap": config.mip_gap,
        "Heuristics": config.heuristics,
        "Cuts": config.cuts,
        "Presolve": config.presolve,
        "Threads": config.threads,
        "TimeLimit": config.time_limit,
        "OutputFlag": config.output_flag,
        "Seed": config.seed,
        "NumericFocus": config.numeric_focus,
    }


STANDARD_PARAMS = _as_gurobi_dictionary(SolverConfig())
DETERMINISTIC_PARAMS = dict(STANDARD_PARAMS)
RISK_AWARE_PARAMS = _as_gurobi_dictionary(SolverConfig(time_limit=7200.0))
RISK_CONSTRAINED_PARAMS = dict(RISK_AWARE_PARAMS)


def get_params(optimization_type: str = "standard") -> dict[str, object]:
    param_sets = {
        "standard": STANDARD_PARAMS,
        "deterministic": DETERMINISTIC_PARAMS,
        "risk_aware": RISK_AWARE_PARAMS,
        "risk_constrained": RISK_CONSTRAINED_PARAMS,
    }
    return dict(param_sets.get(optimization_type, STANDARD_PARAMS))
