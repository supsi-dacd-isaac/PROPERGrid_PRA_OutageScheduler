"""Independent paired out-of-sample validation of two outage schedules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig, RiskConfig, SolverConfig
from .postprocessing import SCOSResult, save_result_bundle
from .risk_measures import paired_cvar_difference_interval, summarise_losses
from .scenario_generation import ScenarioBank


@dataclass
class PairedValidationResult:
    reference: SCOSResult
    cvar: SCOSResult
    paired_losses: pd.DataFrame
    summary: dict[str, Any]


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _subbank(bank: ScenarioBank, labels: list[str]) -> tuple[ScenarioBank, float]:
    mass = float(sum(bank.probabilities[label] for label in labels))
    probabilities = {
        label: bank.probabilities[label] / mass
        for label in labels
    }
    return (
        ScenarioBank(
            demands={label: bank.demands[label] for label in labels},
            probabilities=probabilities,
            seed=bank.seed,
            metadata={**bank.metadata, "batch_labels": labels},
        ),
        mass,
    )


def _weighted_dataframe_sum(items: list[tuple[float, pd.DataFrame]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    result = items[0][1] * items[0][0]
    for mass, frame in items[1:]:
        result = result.add(frame * mass, fill_value=0.0)
    return result


def _combine_batch_results(
    batch_results: list[tuple[float, SCOSResult]],
    original_bank: ScenarioBank,
    beta: float,
    batch_size: int,
) -> SCOSResult:
    if not batch_results:
        raise ValueError("No batch results were provided.")
    first = batch_results[0][1]
    scenario_losses = pd.concat([result.scenario_losses for _, result in batch_results]).loc[original_bank.labels]
    probabilities = pd.Series(original_bank.probabilities, name="probability", dtype=float).loc[scenario_losses.index]
    worst_dns = pd.concat([result.worst_dns for _, result in batch_results]).loc[scenario_losses.index]
    scenario_contingency_energy = pd.concat(
        [result.scenario_contingency_energy for _, result in batch_results]
    ).loc[scenario_losses.index]

    metrics = summarise_losses(
        scenario_losses.to_numpy(),
        beta=beta,
        probabilities=probabilities.to_numpy(),
    ).to_dict()
    runtime = sum(result.solver_summary.get("runtime_sec", 0.0) for _, result in batch_results)
    expected_generation = sum(mass * result.expected_generation for mass, result in batch_results)

    return SCOSResult(
        formulation="validation_expected",
        schedule=first.schedule.copy(),
        start_periods=first.start_periods.copy(),
        scenario_losses=scenario_losses,
        scenario_probabilities=probabilities,
        worst_dns=worst_dns,
        power_generated=_weighted_dataframe_sum(
            [(mass, result.power_generated) for mass, result in batch_results]
        ),
        line_flows=_weighted_dataframe_sum(
            [(mass, result.line_flows) for mass, result in batch_results]
        ),
        base_curtailment=_weighted_dataframe_sum(
            [(mass, result.base_curtailment) for mass, result in batch_results]
        ),
        contingency_curtailment=_weighted_dataframe_sum(
            [(mass, result.contingency_curtailment) for mass, result in batch_results]
        ),
        scenario_contingency_energy=scenario_contingency_energy,
        risk_metrics=metrics,
        maintenance_utility=first.maintenance_utility,
        expected_generation=float(expected_generation),
        solver_summary={
            "status": "BATCHED",
            "runtime_sec": float(runtime),
            "number_of_batches": len(batch_results),
            "batch_statuses": [result.solver_summary.get("status") for _, result in batch_results],
        },
        metadata={
            **first.metadata,
            "validation_mode": "fixed_schedule_batched",
            "validation_batch_size": batch_size,
            "n_scenarios": len(scenario_losses),
        },
        raw_solution=None,
    )


def evaluate_fixed_schedule(
    data: dict[str, Any],
    schedule: pd.DataFrame,
    scenario_bank: ScenarioBank,
    *,
    beta: float,
    model_config: ModelConfig,
    solver_config: SolverConfig,
    batch_size: int | None = None,
) -> SCOSResult:
    """Evaluate one fixed schedule, optionally splitting scenarios into memory-safe batches."""
    from .monolithic_scos import solve_scos

    evaluation_risk = RiskConfig(beta=beta, formulation="expected")
    labels = scenario_bank.labels
    resolved_batch_size = len(labels) if batch_size is None else max(1, min(batch_size, len(labels)))
    batch_results: list[tuple[float, SCOSResult]] = []

    for batch_labels in _chunks(labels, resolved_batch_size):
        bank, mass = _subbank(scenario_bank, batch_labels)
        result = solve_scos(
            data,
            bank,
            risk_config=evaluation_risk,
            model_config=model_config,
            solver_config=solver_config,
            fixed_schedule=schedule,
        )
        batch_results.append((mass, result))

    return _combine_batch_results(
        batch_results,
        scenario_bank,
        beta=beta,
        batch_size=resolved_batch_size,
    )


def validate_schedules(
    data: dict[str, Any],
    reference_schedule: pd.DataFrame,
    cvar_schedule: pd.DataFrame,
    scenario_bank: ScenarioBank,
    *,
    beta: float = 0.95,
    model_config: ModelConfig | None = None,
    solver_config: SolverConfig | None = None,
    batch_size: int | None = 10,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260729,
    output_dir: str | Path | None = None,
) -> PairedValidationResult:
    """
    Evaluate both fixed schedules on the same independent scenario bank.

    Operational variables remain scenario-adaptive, while outage decisions are
    fixed. Validation can be batched because scenario recourse is uncoupled once
    the schedule has been fixed.
    """
    model_config = model_config or ModelConfig()
    solver_config = solver_config or SolverConfig()

    reference = evaluate_fixed_schedule(
        data,
        reference_schedule,
        scenario_bank,
        beta=beta,
        model_config=model_config,
        solver_config=solver_config,
        batch_size=batch_size,
    )
    cvar = evaluate_fixed_schedule(
        data,
        cvar_schedule,
        scenario_bank,
        beta=beta,
        model_config=model_config,
        solver_config=solver_config,
        batch_size=batch_size,
    )

    labels = reference.scenario_losses.index.intersection(cvar.scenario_losses.index)
    if len(labels) != len(reference.scenario_losses) or len(labels) != len(cvar.scenario_losses):
        raise ValueError("Validation results do not share the same scenario labels.")
    probabilities = reference.scenario_probabilities.loc[labels]
    paired = pd.DataFrame(
        {
            "reference_loss_mw_step": reference.scenario_losses.loc[labels],
            "cvar_loss_mw_step": cvar.scenario_losses.loc[labels],
        }
    )
    paired["cvar_minus_reference_mw_step"] = (
        paired["cvar_loss_mw_step"] - paired["reference_loss_mw_step"]
    )
    paired["probability"] = probabilities

    delta, lower, upper = paired_cvar_difference_interval(
        paired["cvar_loss_mw_step"].to_numpy(),
        paired["reference_loss_mw_step"].to_numpy(),
        beta=beta,
        probabilities=probabilities.to_numpy(),
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
    )
    winner = "cvar" if delta < 0 else "reference" if delta > 0 else "tie"
    summary = {
        "beta": beta,
        "winner_by_cvar": winner,
        "reference_metrics": reference.risk_metrics,
        "cvar_metrics": cvar.risk_metrics,
        "cvar_minus_reference": delta,
        "cvar_difference_bootstrap_interval": [lower, upper],
        "mean_paired_difference": float(
            np.average(
                paired["cvar_minus_reference_mw_step"],
                weights=paired["probability"],
            )
        ),
        "probability_cvar_schedule_better": float(
            paired.loc[paired["cvar_minus_reference_mw_step"] < 0, "probability"].sum()
        ),
        "n_scenarios": int(len(paired)),
        "batch_size": batch_size,
        "loss_unit": "MW-step",
        "interpretation": (
            "A negative CVaR difference means the CVaR-informed schedule has a smaller upper-tail loss."
        ),
    }

    validation = PairedValidationResult(
        reference=reference,
        cvar=cvar,
        paired_losses=paired,
        summary=summary,
    )
    if output_dir is not None:
        save_validation_bundle(validation, output_dir)
    return validation


def save_validation_bundle(validation: PairedValidationResult, output_dir: str | Path) -> Path:
    directory = Path(output_dir).expanduser().resolve() / "paired_validation"
    directory.mkdir(parents=True, exist_ok=True)
    save_result_bundle(validation.reference, directory, "reference")
    save_result_bundle(validation.cvar, directory, "cvar")
    validation.paired_losses.to_csv(directory / "paired_scenario_losses.csv")
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(validation.summary, stream, indent=2)
    return directory
