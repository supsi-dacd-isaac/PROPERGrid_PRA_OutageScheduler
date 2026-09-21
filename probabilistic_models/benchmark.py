"""Reproducible out-of-sample comparison of nodal scenario models."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import FloatArray, NodalScenarioModel, as_2d_float
from .data import diagnose_nodal_data
from .metrics import evaluate_ensemble, moving_block_bootstrap_mean_difference
from .nodal import default_model_suite


def _format_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame.loc[:, columns].copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].map(lambda value: f"{value:.5g}")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, divider, *rows])


def _plot_summary(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("primary_rank")
    labels = ordered["model"].str.replace("_", " ")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    axes[0].barh(labels, ordered["energy_score"], color="#2563eb")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Energy score (standardized; lower is better)")
    axes[0].set_title("Joint distribution accuracy")
    axes[0].grid(axis="x", alpha=0.25)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ordered)))
    for color, (_, row) in zip(colors, ordered.iterrows()):
        axes[1].scatter(
            row["spearman_correlation_rmse"],
            row["upper_tail_dependence_rmse_q0.9"],
            color=color,
            s=70,
            label=row["model"].replace("_", " "),
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Spearman-correlation RMSE")
    axes[1].set_ylabel("Upper-tail dependence RMSE")
    axes[1].set_title("Dependence reproduction")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def benchmark_nodal_models(
    values: FloatArray,
    output_dir: str | Path,
    *,
    models: list[NodalScenarioModel] | None = None,
    train_fraction: float = 0.75,
    n_scenarios: int = 1000,
    seed: int = 20260910,
    bootstrap_replicates: int = 1000,
    block_length: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit, score, rank, and compare scenario models on a chronological holdout."""

    data = as_2d_float(values)
    if not 0.5 <= train_fraction < 0.95:
        raise ValueError("train_fraction must be in [0.5, 0.95).")
    split = int(np.floor(train_fraction * data.shape[0]))
    train, test = data[:split], data[split:]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = diagnose_nodal_data(data)
    (output / "diagnostics.json").write_text(
        json.dumps(asdict(diagnostics), indent=2), encoding="utf-8"
    )

    suite = models or default_model_suite()
    records: list[dict[str, float | str]] = []
    energy_series: dict[str, FloatArray] = {}
    for index, model in enumerate(suite):
        started = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scenarios = model.sample(n_scenarios, random_state=seed + 1009 * index)
        sample_seconds = time.perf_counter() - started
        evaluation = evaluate_ensemble(train, test, scenarios)
        energy_series[model.name] = evaluation.energy_score_by_observation
        record: dict[str, float | str] = {
            "model": model.name,
            **evaluation.metrics,
            "fit_seconds": fit_seconds,
            "sample_seconds": sample_seconds,
        }
        if "selected_degrees_of_freedom" in model.metadata_:
            record["selected_degrees_of_freedom"] = float(model.metadata_["selected_degrees_of_freedom"])
        records.append(record)

    summary = pd.DataFrame(records)
    summary["primary_rank"] = summary["energy_score"].rank(method="min").astype(int)
    summary = summary.sort_values(["primary_rank", "model"]).reset_index(drop=True)
    summary.to_csv(output / "benchmark_summary.csv", index=False)

    best_model = str(summary.iloc[0]["model"])
    comparisons = []
    for index, model in enumerate(summary["model"]):
        comparison = moving_block_bootstrap_mean_difference(
            energy_series[model],
            energy_series[best_model],
            block_length=block_length,
            n_bootstrap=bootstrap_replicates,
            random_state=seed + index,
        )
        comparisons.append({"challenger": model, "reference": best_model, **comparison})
    paired = pd.DataFrame(comparisons)
    paired.to_csv(output / "paired_energy_score_bootstrap.csv", index=False)
    _plot_summary(summary, output / "benchmark.png")

    caveat = diagnostics.warning or "No severe rank degeneracy was detected by the automated check."
    table = _format_markdown_table(
        summary,
        [
            "primary_rank",
            "model",
            "energy_score",
            "marginal_crps",
            "variogram_score_p0.5",
            "spearman_correlation_rmse",
            "upper_tail_dependence_rmse_q0.9",
            "central_90pct_coverage",
        ],
    )
    paired_table = _format_markdown_table(
        paired,
        ["challenger", "reference", "mean_difference", "ci95_lower", "ci95_upper"],
    )
    indistinguishable = paired.loc[
        (paired["challenger"] != paired["reference"])
        & (paired["ci95_lower"] <= 0.0)
        & (paired["ci95_upper"] >= 0.0),
        "challenger",
    ].tolist()
    if indistinguishable:
        inference = (
            f"At the 95% moving-block-bootstrap level, {', '.join(indistinguishable)} cannot be "
            f"distinguished from {best_model} on the primary score."
        )
    else:
        inference = f"Every challenger differs from {best_model} at the 95% moving-block-bootstrap level."
    coverage_range = (
        float(summary["central_90pct_coverage"].min()),
        float(summary["central_90pct_coverage"].max()),
    )
    report = f"""# Nodal probabilistic-model benchmark

The comparison uses the first {split:,} observations for fitting and the final
{data.shape[0] - split:,} observations as a chronological holdout. Each fitted
model generates {n_scenarios:,} joint scenarios. The pre-declared primary score
is the multivariate energy score. Marginal CRPS and the variogram score are
proper scoring rules; dependence errors are supplementary diagnostics. All
score calculations exclude constant nodes and standardize active nodes using
training statistics.

{table}

The nominal 90% marginal interval coverage ranges from
{coverage_range[0]:.3f} to {coverage_range[1]:.3f}. The systematic
undercoverage indicates distribution shift between the chronological training
and holdout periods; dependence modelling alone does not correct this marginal
calibration issue.

The paired uncertainty analysis uses a {block_length}-hour moving-block
bootstrap with {bootstrap_replicates:,} replicates. A positive score difference
means that the challenger is worse than the primary-score winner. See
`paired_energy_score_bootstrap.csv` for the confidence intervals. These
intervals quantify test-period sampling uncertainty conditional on the fitted
models and generated ensembles; they do not include parameter-estimation or
finite-ensemble Monte Carlo uncertainty.

{paired_table}

{inference}

## Data adequacy warning

{caveat}

The dataset contains {diagnostics.n_constant_nodes} constant nodes and has
effective rank {diagnostics.effective_rank} across {diagnostics.n_active_nodes}
active nodes. Model rankings therefore describe this particular matrix and
holdout period; they are not evidence that one copula family is universally
superior. A meaningful Gaussian-versus-Student-t tail comparison requires
non-degenerate, stochastic nodal residuals over several seasons or years.
"""
    (output / "benchmark_report.md").write_text(report, encoding="utf-8")
    return summary, paired
