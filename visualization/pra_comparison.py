"""Plots for time-varying and cross-system PRA comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_time_varying_risk(results: pd.DataFrame, output: str | Path) -> Path:
    """Plot overload, voltage, composite, and ROSPRA DNS risk by case and time."""

    frame = results.copy()
    required = {"case", "time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("risk_overload", "Expected overload risk", "%-points"),
        ("risk_undervoltage", "Expected undervoltage risk", "p.u."),
        ("risk_overvoltage", "Expected overvoltage risk", "p.u."),
        ("risk_composite_per_100mw", "Composite risk per 100 MW", "score / 100 MW"),
        ("expected_dns_mwh", "ROSPRA expected DNS", "MWh"),
        ("probability_of_positive_dns", "ROSPRA loss probability", "probability"),
    ]
    available = [item for item in panels if item[0] in frame and frame[item[0]].notna().any()]
    if not available:
        raise ValueError("No recognized risk columns are available for plotting.")

    n_columns = 2
    n_rows = int(np.ceil(len(available) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(11.0, 3.3 * n_rows), squeeze=False)
    for axis, (column, title, ylabel) in zip(axes.flat, available):
        for case_name, subset in frame.groupby("case", sort=False):
            numeric_time = pd.to_numeric(subset["time"], errors="coerce")
            x_values = numeric_time if numeric_time.notna().all() else subset["time"].astype(str)
            axis.plot(
                x_values,
                subset[column],
                marker="o",
                linewidth=1.8,
                label=str(case_name),
            )
        axis.set_title(title)
        axis.set_xlabel("Operating point")
        axis.set_ylabel(ylabel)
        finite = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not finite.empty and (finite >= 0).all():
            axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(available):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Time-varying probabilistic risk assessment", y=0.995, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_case_summary(summary: pd.DataFrame, output: str | Path) -> Path:
    """Plot normalized static-PRA and ROSPRA metrics without mixing their units."""

    frame = summary.copy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_composite_risk_per_100mw", "Static PRA composite risk", "score / 100 MW"),
        ("mean_expected_dns_per_100mw", "ROSPRA expected DNS", "MWh / 100 MW"),
    ]
    available = [item for item in metrics if item[0] in frame and frame[item[0]].notna().any()]
    if not available:
        raise ValueError("No comparable summary metric is available.")

    fig, axes = plt.subplots(1, len(available), figsize=(5.4 * len(available), 4.0), squeeze=False)
    for axis, (column, title, ylabel) in zip(axes.flat, available):
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        axis.bar(frame["case"], values, color="#3569a8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        upper = float(values.max())
        axis.set_ylim(0.0, 1.1 * upper if upper > 0.0 else 1.0)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-system comparison (size-normalized)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
