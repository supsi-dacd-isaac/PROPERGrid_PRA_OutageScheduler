"""Visualisation utilities for paired full-horizon Monte Carlo validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_annual_mc_results(
    summary: Mapping[str, Any],
    annual_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Create publication-oriented comparison figures and return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    labels = ["deterministic", "cvar"]
    display = {"deterministic": "Deterministic", "cvar": "CVaR"}

    # 1. Annual incremental DNS ECDF.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label in labels:
        vals = annual_df.loc[
            annual_df["formulation"] == label,
            "annual_incremental_worst_dns_mwh",
        ].to_numpy(float)
        x, y = _ecdf(vals)
        ax.step(x, y, where="post", label=display[label], linewidth=2)
    ax.set_xlabel("Annual incremental worst-contingency DNS exposure [MWh]")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Annual paired Monte Carlo risk distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out / "01_annual_incremental_dns_ecdf.png"
    _save(fig, path); created.append(path)

    # 2. Exceedance curve.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label in labels:
        vals = np.sort(annual_df.loc[
            annual_df["formulation"] == label,
            "annual_incremental_worst_dns_mwh",
        ].to_numpy(float))
        if vals.size:
            exceed = 1.0 - (np.arange(1, vals.size + 1) - 0.5) / vals.size
            ax.step(vals, exceed, where="post", label=display[label], linewidth=2)
    ax.set_xlabel("Annual incremental worst-contingency DNS exposure [MWh]")
    ax.set_ylabel("Exceedance probability")
    ax.set_title("Annual DNS exceedance curves")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend()
    path = out / "02_annual_dns_exceedance.png"
    _save(fig, path); created.append(path)

    # 3. Paired scatter.
    pivot = annual_df.pivot(index="mc_year", columns="formulation", values="annual_incremental_worst_dns_mwh")
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    if set(labels).issubset(pivot.columns):
        x = pivot["deterministic"].to_numpy(float)
        y = pivot["cvar"].to_numpy(float)
        ax.scatter(x, y, s=34, alpha=0.75)
        lo = min(np.min(x), np.min(y)) if x.size else 0.0
        hi = max(np.max(x), np.max(y)) if x.size else 1.0
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Deterministic annual incremental DNS [MWh]")
    ax.set_ylabel("CVaR annual incremental DNS [MWh]")
    ax.set_title("Paired annual outcomes")
    ax.grid(True, alpha=0.25)
    path = out / "03_paired_annual_dns_scatter.png"
    _save(fig, path); created.append(path)

    # 4. Paired differences.
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if set(labels).issubset(pivot.columns):
        diff = (pivot["cvar"] - pivot["deterministic"]).sort_values()
        ax.bar(np.arange(len(diff)), diff.to_numpy(float))
        ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Monte Carlo year ordered by paired difference")
    ax.set_ylabel("CVaR − deterministic [MWh]")
    ax.set_title("Paired annual DNS differences")
    ax.grid(True, alpha=0.22, axis="y")
    path = out / "04_paired_annual_dns_differences.png"
    _save(fig, path); created.append(path)

    # 5. Summary risk metrics.
    metrics = summary.get("risk_metrics", {})
    names = ["mean", "var", "cvar", "maximum"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(names), dtype=float)
    width = 0.36
    for offset, label in [(-width / 2, "deterministic"), (width / 2, "cvar")]:
        values = [float(metrics.get(label, {}).get(name, 0.0)) for name in names]
        ax.bar(positions + offset, values, width=width, label=display[label])
    ax.set_xticks(positions, ["Mean", "VaR", "CVaR", "Maximum"])
    ax.set_ylabel("Annual incremental DNS exposure [MWh]")
    ax.set_title(f"Annual risk metrics (winner: {summary.get('tail_risk_winner', 'n/a')})")
    ax.grid(True, alpha=0.22, axis="y")
    ax.legend()
    path = out / "05_annual_risk_metrics.png"
    _save(fig, path); created.append(path)

    # 6. Time-step risk frequency aggregated across MC years.
    if not hourly_df.empty:
        grouped = (
            hourly_df.groupby(["time_index", "formulation"], as_index=False)
            .agg(
                probability_positive_incremental_dns=("incremental_worst_dns_mw", lambda x: float(np.mean(np.asarray(x) > 1e-9))),
                mean_incremental_worst_dns_mw=("incremental_worst_dns_mw", "mean"),
            )
        )
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        for label in labels:
            part = grouped[grouped["formulation"] == label]
            ax.plot(part["time_index"], part["probability_positive_incremental_dns"], label=display[label], linewidth=1.5)
        ax.set_xlabel("Planning time-step index")
        ax.set_ylabel("Probability of positive incremental DNS")
        ax.set_title("Risk occurrence across the planning horizon")
        ax.grid(True, alpha=0.22)
        ax.legend()
        path = out / "06_time_step_dns_probability.png"
        _save(fig, path); created.append(path)

    (out / "plot_manifest.json").write_text(
        json.dumps([str(path.name) for path in created], indent=2), encoding="utf-8"
    )
    return created
