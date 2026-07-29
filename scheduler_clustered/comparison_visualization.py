"""Primary deterministic-versus-CVaR comparison visualizations.

Risk conclusions are based only on paired validation results obtained from
identical operating-state samples. Cluster-level ECDFs are deliberately not
used as a formulation-ranking figure because the two schedules generally
produce different cluster boundaries and cluster counts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .results_visualization import FIGURE_DPI, load_results, schedule_matrix


def _step_number(value: Any) -> int:
    text = str(value)
    if text.startswith("step_"):
        text = text.split("_", 1)[1]
    return int(float(text))


def _save(fig: plt.Figure, path: Path, *, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _load_paired(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    result = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("Paired validation JSON must contain an object.")
    return result


def start_comparison(
    deterministic: Mapping[str, Any], cvar: Mapping[str, Any]
) -> pd.DataFrame:
    det = deterministic.get("best_schedule") or {}
    cv = cvar.get("best_schedule") or {}
    outages = sorted(
        set(det) | set(cv),
        key=lambda name: (not str(name).startswith("line_"), str(name)),
    )
    rows: list[dict[str, Any]] = []
    for outage in outages:
        det_label = det.get(outage)
        cv_label = cv.get(outage)
        det_index = _step_number(det_label) if det_label is not None else math.nan
        cv_index = _step_number(cv_label) if cv_label is not None else math.nan
        shift = (
            cv_index - det_index
            if math.isfinite(det_index) and math.isfinite(cv_index)
            else math.nan
        )
        rows.append(
            {
                "outage": str(outage),
                "deterministic_start": det_label,
                "cvar_start": cv_label,
                "deterministic_start_index": det_index,
                "cvar_start_index": cv_index,
                "cvar_minus_deterministic_steps": shift,
                "same_start": bool(math.isfinite(shift) and shift == 0),
            }
        )
    return pd.DataFrame(rows)


def _aligned_schedule_matrices(
    deterministic: Mapping[str, Any], cvar: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    det = schedule_matrix(deterministic)
    cv = schedule_matrix(cvar)
    outages = sorted(
        set(det.index) | set(cv.index),
        key=lambda name: (not str(name).startswith("line_"), str(name)),
    )
    columns = sorted(set(det.columns) | set(cv.columns))
    return (
        det.reindex(index=outages, columns=columns, fill_value=0),
        cv.reindex(index=outages, columns=columns, fill_value=0),
    )


def _evaluation_frame(paired: Mapping[str, Any]) -> pd.DataFrame:
    det_records = {
        str(record["scenario_id"]): record
        for record in paired.get("deterministic_evaluations", [])
    }
    cv_records = {
        str(record["scenario_id"]): record
        for record in paired.get("cvar_evaluations", [])
    }
    probabilities = (
        paired.get("validation_design", {}).get("scenario_probabilities", {})
    )
    scenario_ids = sorted(set(det_records) & set(cv_records))
    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        det = det_records[scenario_id]
        cv = cv_records[scenario_id]
        rows.append(
            {
                "scenario_id": scenario_id,
                "probability": float(probabilities.get(scenario_id, det.get("probability", 0.0))),
                "source_time": det.get("source_time"),
                "total_demand_mw": float(det.get("total_demand", math.nan)),
                "global_multiplier": float(det.get("global_multiplier", math.nan)),
                "deterministic_loss_mw": float(det.get("incremental_maximum_dns", 0.0)),
                "cvar_loss_mw": float(cv.get("incremental_maximum_dns", 0.0)),
                "cvar_minus_deterministic_mw": float(cv.get("incremental_maximum_dns", 0.0))
                - float(det.get("incremental_maximum_dns", 0.0)),
                "deterministic_active_outages": " + ".join(det.get("active_outages", [])),
                "cvar_active_outages": " + ".join(cv.get("active_outages", [])),
                "deterministic_worst_contingency": det.get("candidate_worst_contingency"),
                "cvar_worst_contingency": cv.get("candidate_worst_contingency"),
            }
        )
    return pd.DataFrame(rows)


def _weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(np.sum(ordered_weights))
    if total <= 0.0:
        ordered_weights = np.full(len(values), 1.0 / max(1, len(values)))
    else:
        ordered_weights = ordered_weights / total
    return ordered_values, np.cumsum(ordered_weights)


def plot_schedule_difference(
    deterministic: Mapping[str, Any],
    cvar: Mapping[str, Any],
    path: Path,
    *,
    show: bool = False,
) -> None:
    det, cv = _aligned_schedule_matrices(deterministic, cvar)
    if det.empty:
        return
    difference = cv.to_numpy(dtype=int) - det.to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.45 * len(det) + 1.5)))
    image = ax.imshow(
        difference,
        aspect="auto",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )
    ax.set_yticks(np.arange(len(det.index)))
    ax.set_yticklabels(det.index)
    positions = np.linspace(
        0, max(0, len(det.columns) - 1), min(13, len(det.columns)), dtype=int
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([det.columns[index] for index in positions])
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Planned outage")
    ax.set_title("Schedule difference: CVaR minus deterministic")
    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["Deterministic only", "Same", "CVaR only"])
    _save(fig, path, show=show)


def plot_start_shifts(starts: pd.DataFrame, path: Path, *, show: bool = False) -> None:
    frame = starts.dropna(subset=["cvar_minus_deterministic_steps"]).sort_values(
        "cvar_minus_deterministic_steps"
    )
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42 * len(frame) + 1.5)))
    ax.barh(frame["outage"], frame["cvar_minus_deterministic_steps"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("CVaR start minus deterministic start [steps]")
    ax.set_ylabel("Outage")
    ax.set_title("Outage-start shifts")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path, show=show)


def plot_paired_loss_ecdf(frame: pd.DataFrame, path: Path, *, show: bool = False) -> None:
    if frame.empty:
        return
    weights = frame["probability"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column, label in (
        ("deterministic_loss_mw", "Deterministic"),
        ("cvar_loss_mw", "CVaR"),
    ):
        x, y = _weighted_ecdf(frame[column].to_numpy(dtype=float), weights)
        ax.step(x, y, where="post", label=label)
    ax.set_xlabel("Paired validation incremental maximum DNS [MW]")
    ax.set_ylabel("Weighted cumulative probability")
    ax.set_title("Common-sample loss-distribution comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_risk_metrics(paired: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    comparison = paired["comparison"]
    det = comparison["deterministic"]
    cv = comparison["cvar"]
    labels = ["Expected loss", "VaR", "CVaR", "Maximum loss"]
    det_values = [det["expected_loss"], det["var"], det["cvar"], det["maximum_loss"]]
    cv_values = [cv["expected_loss"], cv["var"], cv["cvar"], cv["maximum_loss"]]
    positions = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    det_bars = ax.bar(positions - width / 2, det_values, width, label="Deterministic")
    cv_bars = ax.bar(positions + width / 2, cv_values, width, label="CVaR")
    ax.bar_label(det_bars, fmt="%.3g", padding=3)
    ax.bar_label(cv_bars, fmt="%.3g", padding=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Incremental DNS [MW]")
    winner = comparison.get("tail_risk_winner", "undetermined")
    ax.set_title(f"Paired validation risk metrics; tail-risk winner: {winner}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path, show=show)


def plot_paired_scatter(frame: pd.DataFrame, path: Path, *, show: bool = False) -> None:
    if frame.empty:
        return
    x = frame["deterministic_loss_mw"].to_numpy(dtype=float)
    y = frame["cvar_loss_mw"].to_numpy(dtype=float)
    limit = max(float(np.max(x, initial=0.0)), float(np.max(y, initial=0.0)), 1e-9)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=45)
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", label="Equal loss")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Deterministic loss [MW]")
    ax.set_ylabel("CVaR-schedule loss [MW]")
    ax.set_title("Paired sample-by-sample comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_paired_differences(frame: pd.DataFrame, path: Path, *, show: bool = False) -> None:
    if frame.empty:
        return
    ordered = frame.sort_values("cvar_minus_deterministic_mw").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(np.arange(len(ordered)), ordered["cvar_minus_deterministic_mw"])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Validation samples ordered by paired loss difference")
    ax.set_ylabel("CVaR minus deterministic loss [MW]")
    ax.set_title("Paired loss differences; negative values favour CVaR")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path, show=show)


def plot_upper_tail(frame: pd.DataFrame, paired: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    if frame.empty:
        return
    alpha = float(paired["comparison"]["alpha"])
    weights = frame["probability"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column, label in (
        ("deterministic_loss_mw", "Deterministic"),
        ("cvar_loss_mw", "CVaR"),
    ):
        x, cdf = _weighted_ecdf(frame[column].to_numpy(dtype=float), weights)
        mask = cdf >= alpha - 1e-12
        ax.step(cdf[mask], x[mask], where="post", label=label)
    ax.axvline(alpha, linestyle="--", label=f"VaR level {alpha:.2f}")
    ax.set_xlabel("Weighted cumulative probability")
    ax.set_ylabel("Incremental DNS [MW]")
    ax.set_title("Upper-tail quantile comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def generate_unambiguous_comparison(
    deterministic_path: str | Path,
    cvar_path: str | Path,
    paired_validation_path: str | Path,
    *,
    output_dir: str | Path,
    show: bool = False,
) -> Path:
    deterministic = load_results(deterministic_path)
    cvar = load_results(cvar_path)
    paired = _load_paired(paired_validation_path)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    starts = start_comparison(deterministic, cvar)
    frame = _evaluation_frame(paired)
    comparison = paired["comparison"]

    starts.to_csv(directory / "outage_start_comparison.csv", index=False)
    frame.to_csv(directory / "paired_sample_losses.csv", index=False)
    risk_frame = pd.DataFrame(
        [
            {
                "formulation": formulation,
                "expected_loss_mw": comparison[formulation]["expected_loss"],
                "var_mw": comparison[formulation]["var"],
                "cvar_mw": comparison[formulation]["cvar"],
                "maximum_loss_mw": comparison[formulation]["maximum_loss"],
                "probability_positive_loss": comparison[formulation]["probability_positive_loss"],
            }
            for formulation in ("deterministic", "cvar")
        ]
    )
    risk_frame.to_csv(directory / "paired_risk_metric_summary.csv", index=False)

    schedule_statistics = {
        "number_of_outages": int(len(starts)),
        "same_start_count": int(starts["same_start"].sum()) if not starts.empty else 0,
        "shifted_outage_count": int((~starts["same_start"]).sum()) if not starts.empty else 0,
        "mean_absolute_start_shift_steps": float(
            starts["cvar_minus_deterministic_steps"].dropna().abs().mean()
        ) if not starts.empty else math.nan,
        "maximum_absolute_start_shift_steps": float(
            starts["cvar_minus_deterministic_steps"].dropna().abs().max()
        ) if not starts.empty else math.nan,
    }
    summary = {
        "risk_comparison": comparison,
        "schedule_statistics": schedule_statistics,
        "interpretation_rule": (
            "The formulation with the lower paired-validation CVaR is declared "
            "less tail-risky. Formulation-specific optimisation scores and "
            "unpaired cluster-severity distributions are not used for this verdict."
        ),
    }
    (directory / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (directory / "CONCLUSION.txt").write_text(
        comparison["conclusion"]
        + "\nTail-risk winner: "
        + comparison["tail_risk_winner"]
        + "\n",
        encoding="utf-8",
    )

    plot_schedule_difference(
        deterministic, cvar, directory / "01_schedule_difference.png", show=show
    )
    plot_start_shifts(starts, directory / "02_outage_start_shifts.png", show=show)
    plot_paired_loss_ecdf(frame, directory / "03_paired_loss_ecdf.png", show=show)
    plot_risk_metrics(paired, directory / "04_paired_risk_metrics.png", show=show)
    plot_paired_scatter(frame, directory / "05_paired_loss_scatter.png", show=show)
    plot_paired_differences(frame, directory / "06_paired_loss_differences.png", show=show)
    plot_upper_tail(frame, paired, directory / "07_upper_tail_comparison.png", show=show)
    return directory
