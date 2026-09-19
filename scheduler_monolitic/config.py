"""Configuration objects for the restructured monolithic SCOS package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SolverConfig:
    """Gurobi settings shared by all formulations."""

    mip_gap: float = 0.01
    time_limit: float = 3600.0
    threads: int = 8
    output_flag: int = 1
    seed: int = 42
    numeric_focus: int = 1
    presolve: int = 2
    cuts: int = 2
    heuristics: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= self.mip_gap <= 1.0:
            raise ValueError("mip_gap must lie in [0, 1].")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive.")
        if self.threads < 1:
            raise ValueError("threads must be at least one.")

    def apply(self, model: Any) -> None:
        """Apply parameters to a Gurobi model."""
        self.validate()
        params = {
            "MIPGap": self.mip_gap,
            "TimeLimit": self.time_limit,
            "Threads": self.threads,
            "OutputFlag": self.output_flag,
            "Seed": self.seed,
            "NumericFocus": self.numeric_focus,
            "Presolve": self.presolve,
            "Cuts": self.cuts,
            "Heuristics": self.heuristics,
        }
        for key, value in params.items():
            model.setParam(key, value)


@dataclass(frozen=True)
class ScenarioConfig:
    """Demand-scenario generation settings."""

    n_scenarios: int = 20
    seed: int = 42
    sampling_scheme: str = "stratified_tail"
    sigma_global: float = 0.05
    sigma_local: float = 0.02
    temporal_correlation: float = 0.80
    tail_fraction: float = 0.25

    def validate(self) -> None:
        if self.n_scenarios < 1:
            raise ValueError("n_scenarios must be positive.")
        if self.sampling_scheme not in {"random", "stratified_tail"}:
            raise ValueError("sampling_scheme must be 'random' or 'stratified_tail'.")
        if self.sigma_global < 0 or self.sigma_local < 0:
            raise ValueError("Demand standard deviations must be non-negative.")
        if not 0.0 <= self.temporal_correlation < 1.0:
            raise ValueError("temporal_correlation must lie in [0, 1).")
        if not 0.0 < self.tail_fraction < 1.0:
            raise ValueError("tail_fraction must lie in (0, 1).")


@dataclass(frozen=True)
class RiskConfig:
    """Risk functional and comparison settings."""

    beta: float = 0.95
    formulation: str = "expected"  # expected | cvar | weighted
    expected_weight: float = 1.0
    cvar_weight: float = 10.0
    generation_weight: float = 1.0e-3
    utility_weight: float = 1.0e-3
    utility_tolerance: float = 0.0
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 20260729

    def validate(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (0, 1).")
        if self.formulation not in {"expected", "cvar", "weighted"}:
            raise ValueError("formulation must be expected, cvar, or weighted.")
        if min(self.expected_weight, self.cvar_weight, self.generation_weight, self.utility_weight) < 0:
            raise ValueError("Objective weights must be non-negative.")
        if self.utility_tolerance < 0:
            raise ValueError("utility_tolerance must be non-negative.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative.")


@dataclass(frozen=True)
class ModelConfig:
    """Physical and structural settings of the SCOS formulation."""

    use_dc_power_flow: bool = True
    include_base_state: bool = True
    corrective_redispatch_fraction: float = 0.20
    contingency_top_k: int | None = None
    spill_penalty: float = 1.0e-4
    angle_bound: float = 45.0
    store_full_solution: bool = False
    max_estimated_variables: int | None = 5_000_000
    allow_large_model: bool = False
    name_operational_constraints: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.corrective_redispatch_fraction <= 1.0:
            raise ValueError("corrective_redispatch_fraction must lie in [0, 1].")
        if self.contingency_top_k is not None and self.contingency_top_k < 1:
            raise ValueError("contingency_top_k must be positive or None.")
        if self.angle_bound <= 0:
            raise ValueError("angle_bound must be positive.")
        if self.max_estimated_variables is not None and self.max_estimated_variables < 1:
            raise ValueError("max_estimated_variables must be positive or None.")


@dataclass(frozen=True)
class OutputConfig:
    """Output paths relative to the repository root."""

    root: Path | None = None
    figures_subdir: str = "figures"

    def resolve_root(self, package_file: str | Path) -> Path:
        if self.root is not None:
            return Path(self.root).expanduser().resolve()
        repository_root = Path(package_file).resolve().parents[1]
        return repository_root / "outputs" / "schedule_results" / "monolitic_scheduler"


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert supported dataclasses to JSON-ready dictionaries."""
    result = asdict(value)
    for key, item in list(result.items()):
        if isinstance(item, Path):
            result[key] = str(item)
    return result


def update_dataclass(instance: Any, overrides: Mapping[str, Any]) -> Any:
    """Return a dataclass copy with selected fields replaced."""
    values = dataclass_to_dict(instance)
    unknown = set(overrides) - set(values)
    if unknown:
        raise KeyError(f"Unknown configuration fields: {sorted(unknown)}")
    values.update(overrides)
    return type(instance)(**values)
