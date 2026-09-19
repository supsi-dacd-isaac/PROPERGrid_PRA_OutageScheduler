"""Compact publication-oriented comparison plots for monolithic SCOS results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .postprocessing import SCOSResult
from .schedule_validation import PairedValidationResult


def _step_numbers(columns: Iterable[object]) -> np.ndarray:
    values = []
    for index, column in enumerate(columns):
        if isinstance(column, str) and column.startswith("step_"):
            values.append(int(column[5:]))
        else:
            try:
                values.append(int(column))
            except (TypeError, ValueError):
                values.append(index)
    return np.asarray(values)


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_schedule_comparison(
    reference: SCOSResult,
    cvar: SCOSResult,
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    outages = list(reference.schedule.index)
    if outages != list(cvar.schedule.index):
        raise ValueError("Schedules do not contain the same outage rows.")

    reference_schedule = reference.schedule.loc[outages].to_numpy(dtype=float)
    cvar_schedule = cvar.schedule.loc[outages].to_numpy(dtype=float)
    difference = cvar_schedule - reference_schedule
    time = _step_numbers(reference.schedule.columns)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    extent = [time.min(), time.max() + 1, len(outages) - 0.5, -0.5]

    image = axes[0, 0].imshow(reference_schedule, aspect="auto", vmin=0, vmax=1, extent=extent)
    axes[0, 0].set_title("Risk-neutral schedule")
    axes[0, 0].set_yticks(range(len(outages)), outages)
    axes[0, 0].set_ylabel("Outage")
    fig.colorbar(image, ax=axes[0, 0], label="Active status")

    image = axes[0, 1].imshow(cvar_schedule, aspect="auto", vmin=0, vmax=1, extent=extent)
    axes[0, 1].set_title("CVaR-informed schedule")
    axes[0, 1].set_yticks(range(len(outages)), outages)
    fig.colorbar(image, ax=axes[0, 1], label="Active status")

    image = axes[1, 0].imshow(difference, aspect="auto", vmin=-1, vmax=1, extent=extent, cmap="coolwarm")
    axes[1, 0].set_title("Schedule difference: CVaR − risk-neutral")
    axes[1, 0].set_yticks(range(len(outages)), outages)
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Outage")
    fig.colorbar(image, ax=axes[1, 0], label="Status difference")

    axes[1, 1].step(time, reference.schedule.sum(axis=0), where="post", label="Risk-neutral")
    axes[1, 1].step(time, cvar.schedule.sum(axis=0), where="post", linestyle="--", label="CVaR")
    axes[1, 1].set_title("Simultaneous planned outages")
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Number of active outages")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Outage-schedule comparison")
    _save(fig, directory, "comparison_01_outage_schedules")

    starts = pd.DataFrame(
        {
            "Risk-neutral": reference.start_periods,
            "CVaR": cvar.start_periods,
        }
    )
    shift = starts["CVaR"] - starts["Risk-neutral"]
    positions = np.arange(len(starts))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(positions - width / 2, starts["Risk-neutral"], width, label="Risk-neutral")
    axes[0].bar(positions + width / 2, starts["CVaR"], width, label="CVaR")
    axes[0].set_xticks(positions, starts.index, rotation=45, ha="right")
    axes[0].set_ylabel("Start step")
    axes[0].set_title("Outage start periods")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(positions, shift)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_xticks(positions, starts.index, rotation=45, ha="right")
    axes[1].set_ylabel("Shift [steps]")
    axes[1].set_title("Start shift: CVaR − risk-neutral")
    axes[1].grid(axis="y", alpha=0.3)
    _save(fig, directory, "comparison_02_outage_start_shifts")
    starts.assign(shift_cvar_minus_reference=shift).to_csv(directory / "comparison_outage_start_shifts.csv")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def plot_tail_risk_comparison(
    validation: PairedValidationResult,
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    paired = validation.paired_losses
    reference_losses = paired["reference_loss_mw_step"].to_numpy(dtype=float)
    cvar_losses = paired["cvar_loss_mw_step"].to_numpy(dtype=float)
    beta = float(validation.summary["beta"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for losses, label, linestyle in [
        (reference_losses, "Risk-neutral", "-"),
        (cvar_losses, "CVaR", "--"),
    ]:
        x, y = _ecdf(losses)
        axes[0, 0].step(x, y, where="post", label=label, linestyle=linestyle)
        survival = np.maximum(1.0 - y + 1.0 / len(y), 1.0 / len(y))
        axes[0, 1].step(x, survival, where="post", label=label, linestyle=linestyle)

    reference_var = validation.reference.risk_metrics["var"]
    cvar_var = validation.cvar.risk_metrics["var"]
    axes[0, 0].axvspan(min(reference_var, cvar_var), max(reference_losses.max(), cvar_losses.max()), alpha=0.08)
    axes[0, 0].set_title(f"Out-of-sample loss ECDF; shaded upper tail (β={beta:.2f})")
    axes[0, 0].set_xlabel("Schedule loss [MW-step]")
    axes[0, 0].set_ylabel("Empirical probability")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Loss exceedance probability")
    axes[0, 1].set_xlabel("Schedule loss [MW-step]")
    axes[0, 1].set_ylabel("P(L ≥ x)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    metrics = ["expected", "var", "cvar", "q99", "maximum"]
    labels = ["Mean", f"VaR{int(beta * 100)}", f"CVaR{int(beta * 100)}", "Q99", "Maximum"]
    positions = np.arange(len(metrics))
    width = 0.38
    reference_values = [validation.reference.risk_metrics[key] for key in metrics]
    cvar_values = [validation.cvar.risk_metrics[key] for key in metrics]
    axes[1, 0].bar(positions - width / 2, reference_values, width, label="Risk-neutral")
    axes[1, 0].bar(positions + width / 2, cvar_values, width, label="CVaR")
    axes[1, 0].set_xticks(positions, labels, rotation=20)
    axes[1, 0].set_ylabel("Loss [MW-step]")
    axes[1, 0].set_title("Out-of-sample risk metrics")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    differences = np.sort(paired["cvar_minus_reference_mw_step"].to_numpy(dtype=float))
    axes[1, 1].bar(np.arange(len(differences)), differences)
    axes[1, 1].axhline(0, linewidth=1)
    axes[1, 1].set_title("Paired scenario difference: CVaR − risk-neutral")
    axes[1, 1].set_xlabel("Scenarios ordered by paired difference")
    axes[1, 1].set_ylabel("Loss difference [MW-step]")
    axes[1, 1].grid(axis="y", alpha=0.3)

    delta = validation.summary["cvar_minus_reference"]
    lower, upper = validation.summary["cvar_difference_bootstrap_interval"]
    fig.suptitle(
        f"Paired tail-risk validation: ΔCVaR={delta:.3f} MW-step "
        f"(bootstrap interval [{lower:.3f}, {upper:.3f}])"
    )
    _save(fig, directory, "comparison_03_tail_risk")

    summary_table = pd.DataFrame(
        {
            "metric": labels,
            "risk_neutral": reference_values,
            "cvar": cvar_values,
            "cvar_minus_risk_neutral": np.asarray(cvar_values) - np.asarray(reference_values),
        }
    )
    summary_table.to_csv(directory / "comparison_risk_metrics.csv", index=False)


def plot_tail_contingency_contributions(
    validation: PairedValidationResult,
    output_dir: str | Path,
    top_n: int = 10,
) -> None:
    directory = Path(output_dir)
    beta = float(validation.summary["beta"])

    def conditional_tail_contribution(result: SCOSResult) -> pd.Series:
        threshold = result.risk_metrics["var"]
        tail_labels = result.scenario_losses[result.scenario_losses >= threshold - 1.0e-9].index
        if len(tail_labels) == 0:
            return pd.Series(dtype=float)
        probabilities = result.scenario_probabilities.loc[tail_labels]
        probabilities = probabilities / probabilities.sum()
        values = result.scenario_contingency_energy.loc[tail_labels]
        return values.mul(probabilities, axis=0).sum(axis=0)

    reference = conditional_tail_contribution(validation.reference)
    cvar = conditional_tail_contribution(validation.cvar)
    combined = pd.DataFrame({"Risk-neutral": reference, "CVaR": cvar}).fillna(0.0)
    ranking = combined.max(axis=1).sort_values(ascending=False).head(top_n).index
    combined = combined.loc[ranking]

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    positions = np.arange(len(combined))
    width = 0.38
    ax.bar(positions - width / 2, combined["Risk-neutral"], width, label="Risk-neutral")
    ax.bar(positions + width / 2, combined["CVaR"], width, label="CVaR")
    ax.set_xticks(positions, combined.index, rotation=45, ha="right")
    ax.set_ylabel("Conditional tail curtailment [MW-step]")
    ax.set_title(f"Contingency curtailment exposure in each schedule's upper {100 * (1 - beta):.1f}% tail")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, directory, "comparison_04_tail_contingencies")
    combined.to_csv(directory / "comparison_tail_contingency_contributions.csv")


def plot_operational_comparison(
    reference: SCOSResult,
    cvar: SCOSResult,
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    time = _step_numbers(reference.power_generated.columns)
    reference_generation = reference.power_generated.sum(axis=0).to_numpy(dtype=float)
    cvar_generation = cvar.power_generated.sum(axis=0).to_numpy(dtype=float)

    reference_dns = reference.contingency_curtailment.max(axis=0).to_numpy(dtype=float)
    cvar_dns = cvar.contingency_curtailment.max(axis=0).to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    axes[0].plot(time, reference_generation, label="Risk-neutral")
    axes[0].plot(time, cvar_generation, linestyle="--", label="CVaR")
    axes[0].set_title("Expected preventive generation")
    axes[0].set_ylabel("Generation [MW]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, reference_dns, label="Risk-neutral")
    axes[1].plot(time, cvar_dns, linestyle="--", label="CVaR")
    axes[1].set_title("Expected worst N−1 curtailment by time step")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Load shed [MW]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _save(fig, directory, "comparison_05_operational_profiles")


def plot_comparison_suite(
    training_reference: SCOSResult,
    training_cvar: SCOSResult,
    validation: PairedValidationResult,
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plot_schedule_comparison(training_reference, training_cvar, directory)
    plot_tail_risk_comparison(validation, directory)
    plot_tail_contingency_contributions(validation, directory)
    plot_operational_comparison(validation.reference, validation.cvar, directory)
