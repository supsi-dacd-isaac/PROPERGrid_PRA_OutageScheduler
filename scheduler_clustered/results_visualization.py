"""Visualization and tabular post-processing for clustered scheduler results.

The module reads the JSON files produced by either the deterministic clustered
scheduler or the clustered CVaR scheduler.  It creates publication-quality,
stand-alone figures and CSV summaries without requiring the original Gurobi
model or the PROPER input data.

Examples
--------
python -m scheduler_clustered.results_visualization clustered_deterministic_results.json
python -m scheduler_clustered.results_visualization clustered_cvar_results.json --show
python -m scheduler_clustered.results_visualization deterministic.json --compare cvar.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DPI = 180


def _step_number(label: str | int | float) -> int:
    if isinstance(label, (int, np.integer)):
        return int(label)
    text = str(label)
    if text.startswith("step_"):
        return int(text.split("_", 1)[1])
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Cannot infer a time index from {label!r}.") from exc


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_results(path: str | Path) -> dict[str, Any]:
    """Load one clustered-result JSON file.

    Python's JSON decoder accepts ``Infinity`` values written by the current
    deterministic scheduler.  They are retained in memory and sanitized only
    where a finite plotting value is required.
    """
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with result_path.open("r", encoding="utf-8") as stream:
        results = json.load(stream)
    if not isinstance(results, dict):
        raise TypeError("The result JSON root must be an object.")
    return results


def _candidate_signature(candidate: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in candidate.get("start_times", {}).items()))


def best_candidate(results: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the candidate corresponding to ``best_schedule``/``best_score``."""
    candidates = list(results.get("candidates", []))
    if not candidates:
        raise ValueError("The result JSON contains no candidate records.")

    schedule = results.get("best_schedule")
    if isinstance(schedule, Mapping):
        signature = tuple(sorted((str(k), str(v)) for k, v in schedule.items()))
        matches = [c for c in candidates if _candidate_signature(c) == signature]
        if matches:
            return max(matches, key=lambda c: _finite(c.get("score"), -np.inf))

    best_score = _finite(results.get("best_score"), np.nan)
    if math.isfinite(best_score):
        return min(
            candidates,
            key=lambda c: abs(_finite(c.get("score"), -np.inf) - best_score),
        )
    return max(candidates, key=lambda c: _finite(c.get("score"), -np.inf))


def candidate_clusters(results: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return full-validation clusters when available, otherwise best-candidate clusters."""
    validation = results.get("full_validation_clusters")
    if isinstance(validation, list) and validation:
        return validation
    return list(best_candidate(results).get("clusters", []))


def _ordered_times(results: Mapping[str, Any]) -> list[str]:
    active = results.get("best_active_outages")
    if isinstance(active, Mapping) and active:
        return sorted((str(t) for t in active), key=_step_number)

    clusters = candidate_clusters(results)
    max_index = -1
    for record in clusters:
        cluster = record.get("cluster", {})
        max_index = max(max_index, int(cluster.get("end_index", 0)) - 1)
    if max_index < 0:
        schedule = results.get("best_schedule", {})
        max_index = max((_step_number(v) for v in schedule.values()), default=0)
    return [f"step_{index}" for index in range(max_index + 1)]


def schedule_matrix(results: Mapping[str, Any]) -> pd.DataFrame:
    times = _ordered_times(results)
    active = results.get("best_active_outages", {})
    outage_names: set[str] = set(results.get("best_schedule", {}).keys())
    if isinstance(active, Mapping):
        for outages in active.values():
            outage_names.update(str(outage) for outage in outages)
    outages = sorted(outage_names, key=lambda name: (not name.startswith("line_"), name))
    matrix = pd.DataFrame(0, index=outages, columns=[_step_number(t) for t in times], dtype=int)
    if isinstance(active, Mapping):
        for time, names in active.items():
            column = _step_number(time)
            if column not in matrix.columns:
                continue
            for outage in names:
                if str(outage) in matrix.index:
                    matrix.loc[str(outage), column] = 1
    else:
        for record in candidate_clusters(results):
            cluster = record.get("cluster", {})
            start = int(cluster.get("start_index", 0))
            end = int(cluster.get("end_index", start))
            for outage in cluster.get("active_outages", []):
                if str(outage) in matrix.index:
                    matrix.loc[str(outage), start : end - 1] = 1
    return matrix


def cluster_summary(results: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in candidate_clusters(results):
        cluster = record.get("cluster", {})
        representative = record.get("representative", {})
        candidate = record.get("candidate", {})
        baseline = record.get("baseline", {})
        contingencies = list(candidate.get("contingencies", []))
        total_increment = _finite(record.get("total_incremental_dns"), 0.0)
        risk = record.get("risk") or {}
        mean_increment = (
            _finite(risk.get("expected_loss"), np.nan)
            if risk
            else total_increment / max(1, len(contingencies) + 1)
        )
        rows.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "start_index": int(cluster.get("start_index", 0)),
                "end_index": int(cluster.get("end_index", 0)),
                "duration_steps": int(cluster.get("duration_steps", 0)),
                "active_outages": " + ".join(cluster.get("active_outages", [])),
                "outage_count": len(cluster.get("active_outages", [])),
                "representative_time": representative.get("time"),
                "representative_total_demand_mw": _finite(representative.get("total_demand")),
                "screening_score": _finite(representative.get("screening_score")),
                "worst_screened_contingency": representative.get("worst_screened_contingency"),
                "candidate_max_dns_mw": _finite(candidate.get("maximum_dns"), 0.0),
                "candidate_total_dns_mw": _finite(candidate.get("total_dns"), 0.0),
                "baseline_max_dns_mw": _finite(baseline.get("maximum_dns"), 0.0),
                "baseline_total_dns_mw": _finite(baseline.get("total_dns"), 0.0),
                "incremental_max_dns_mw": _finite(record.get("maximum_incremental_dns"), 0.0),
                "incremental_total_dns_mw": total_increment,
                "incremental_mean_contingency_dns_mw": mean_increment,
                "normalized_severity": _finite(record.get("normalized_severity"), 0.0),
                "worst_contingency": candidate.get("worst_contingency"),
                "budget_feasible": bool(record.get("budget_feasible", False)),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("start_index").reset_index(drop=True)
    return frame


def candidate_summary(results: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in results.get("candidates", []):
        risk = candidate.get("risk") or {}
        rows.append(
            {
                "iteration": int(candidate.get("iteration", len(rows) + 1)),
                "maintenance_utility": _finite(candidate.get("maintenance_utility")),
                "deterministic_security_cost": _finite(candidate.get("deterministic_security_cost")),
                "expected_dns_mw": _finite(risk.get("expected_loss")),
                "var_dns_mw": _finite(risk.get("var")),
                "cvar_dns_mw": _finite(risk.get("cvar")),
                "score": _finite(candidate.get("score")),
                "budget_feasible": bool(candidate.get("budget_feasible", False)),
                "number_of_clusters": len(candidate.get("clusters", [])),
            }
        )
    return pd.DataFrame(rows).sort_values("iteration") if rows else pd.DataFrame()


def _save(fig: plt.Figure, output_path: Path, *, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_schedule(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    matrix = schedule_matrix(results)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.42 * len(matrix.index) + 1.5)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    tick_step = max(1, len(matrix.columns) // 12)
    ticks = np.arange(0, len(matrix.columns), tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([matrix.columns[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Planned outage")
    ax.set_title("Optimized outage schedule")
    ax.grid(False)
    _save(fig, output_path, show=show)


def plot_schedule_gantt(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    matrix = schedule_matrix(results)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.45 * len(matrix.index) + 1.5)))
    for row_index, outage in enumerate(matrix.index):
        active = matrix.loc[outage].to_numpy(dtype=int)
        starts = np.flatnonzero(np.diff(np.r_[0, active]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[active, 0]) == -1) + 1
        intervals = [(int(start), int(end - start)) for start, end in zip(starts, ends)]
        if intervals:
            ax.broken_barh(intervals, (row_index - 0.36, 0.72))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_ylim(-0.8, len(matrix.index) - 0.2)
    ax.set_xlim(0, max(matrix.columns) + 1)
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Planned outage")
    ax.set_title("Optimized outage schedule (Gantt view)")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, output_path, show=show)


def plot_active_outage_count(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    matrix = schedule_matrix(results)
    if matrix.empty:
        return
    count = matrix.sum(axis=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(count.index, count.values, where="post")
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Concurrent planned outages")
    ax.set_title("Maintenance concurrency over the planning horizon")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    _save(fig, output_path, show=show)


def plot_dns_by_cluster(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = cluster_summary(results)
    if frame.empty:
        return
    x = np.arange(len(frame))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, 0.95 * len(frame)), 5.5))
    ax.bar(x - width / 2, frame["incremental_max_dns_mw"], width, label="Maximum incremental DNS")
    ax.bar(x + width / 2, frame["incremental_mean_contingency_dns_mw"], width, label="Mean incremental DNS across states")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["cluster_id"], rotation=45, ha="right")
    ax.set_xlabel("Outage cluster")
    ax.set_ylabel("Incremental DNS [MW]")
    ax.set_title("Incremental DNS by outage cluster")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path, show=show)


def plot_cluster_severity_timeline(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = cluster_summary(results)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    for _, row in frame.iterrows():
        start = float(row["start_index"])
        duration = float(row["duration_steps"])
        severity = float(row["incremental_max_dns_mw"])
        ax.bar(start, severity, width=duration, align="edge", edgecolor="black", linewidth=0.4)
        if severity > 0:
            ax.text(start + duration / 2, severity, str(row["cluster_id"]), ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Maximum incremental DNS [MW]")
    ax.set_title("Security severity over the outage-cluster timeline")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path, show=show)


def plot_critical_clusters(
    results: Mapping[str, Any], output_path: Path, *, show: bool = False, top_n: int = 10
) -> None:
    frame = cluster_summary(results)
    if frame.empty:
        return
    ranked = frame.sort_values(
        ["normalized_severity", "incremental_max_dns_mw"], ascending=False
    ).head(top_n)
    labels = [f"{row.cluster_id}: {row.active_outages}" for row in ranked.itertuples()]
    values = ranked["normalized_severity"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.55 * len(ranked) + 1.5)))
    positions = np.arange(len(ranked))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Duration-weighted normalized severity")
    ax.set_title("Most critical outage clusters")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, output_path, show=show)


def _pair_severity(frame: pd.DataFrame) -> pd.DataFrame:
    names: set[str] = set()
    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in frame.itertuples():
        outages = [part.strip() for part in str(row.active_outages).split("+") if part.strip()]
        names.update(outages)
        if len(outages) == 1:
            values[(outages[0], outages[0])].append(float(row.normalized_severity))
        else:
            for first_index in range(len(outages)):
                for second_index in range(first_index + 1, len(outages)):
                    first, second = sorted((outages[first_index], outages[second_index]))
                    values[(first, second)].append(float(row.normalized_severity))
    ordered = sorted(names, key=lambda name: (not name.startswith("line_"), name))
    matrix = pd.DataFrame(0.0, index=ordered, columns=ordered)
    for (first, second), observations in values.items():
        value = float(np.mean(observations))
        matrix.loc[first, second] = value
        matrix.loc[second, first] = value
    return matrix


def plot_outage_interaction_matrix(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = cluster_summary(results)
    matrix = _pair_severity(frame)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(matrix)), max(7, 0.58 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(), aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Observed outage-interaction severity")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean normalized cluster severity")
    ax.grid(False)
    _save(fig, output_path, show=show)


def plot_worst_contingencies(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = cluster_summary(results)
    frame = frame[frame["worst_contingency"].notna()]
    if frame.empty:
        return
    severity = frame.groupby("worst_contingency")["incremental_max_dns_mw"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.45 * len(severity) + 1.5)))
    positions = np.arange(len(severity))
    ax.barh(positions, severity.values)
    ax.set_yticks(positions)
    ax.set_yticklabels(severity.index)
    ax.set_xlabel("Sum of associated maximum incremental DNS [MW]")
    ax.set_title("Critical N-1 contingencies across the best schedule")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, output_path, show=show)


def plot_optimization_convergence(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = candidate_summary(results)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(frame["iteration"], frame["score"], marker="o", label="Candidate score")
    ax.plot(frame["iteration"], frame["maintenance_utility"], marker="s", label="Maintenance utility")
    finite_cvar = frame["cvar_dns_mw"].notna().any()
    if finite_cvar:
        ax2 = ax.twinx()
        ax2.plot(frame["iteration"], frame["cvar_dns_mw"], marker="^", linestyle="--", label="CVaR")
        ax2.set_ylabel("CVaR of incremental DNS [MW]")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend()
    ax.set_xlabel("Outer-loop iteration")
    ax.set_ylabel("Optimization score / utility")
    ax.set_title("Outer-loop convergence")
    ax.grid(alpha=0.3)
    _save(fig, output_path, show=show)


def _scenario_losses(results: Mapping[str, Any]) -> pd.DataFrame:
    losses = results.get("best_scenario_losses")
    probabilities = results.get("best_scenario_probabilities", {})
    tail_ids = set((results.get("best_risk") or {}).get("tail_ids", []))
    if not isinstance(losses, Mapping):
        return pd.DataFrame()
    rows = []
    for scenario_id, loss in losses.items():
        rows.append(
            {
                "scenario_id": str(scenario_id),
                "loss_mw": _finite(loss, 0.0),
                "probability": _finite(probabilities.get(scenario_id, 1.0), 1.0),
                "tail": str(scenario_id) in tail_ids,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        total = frame["probability"].sum()
        if total > 0:
            frame["probability"] /= total
        frame = frame.sort_values("loss_mw").reset_index(drop=True)
    return frame


def plot_cvar_distribution(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = _scenario_losses(results)
    risk = results.get("best_risk") or {}
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    non_tail = frame[~frame["tail"]]
    tail = frame[frame["tail"]]
    ax.scatter(non_tail["loss_mw"], non_tail["probability"], label="Non-tail scenarios")
    if not tail.empty:
        ax.scatter(tail["loss_mw"], tail["probability"], marker="x", s=70, label="CVaR-tail scenarios")
    var = _finite(risk.get("var"))
    cvar = _finite(risk.get("cvar"))
    if math.isfinite(var):
        ax.axvline(var, linestyle="--", label=f"VaR = {var:.2f} MW")
    if math.isfinite(cvar):
        ax.axvline(cvar, linestyle=":", label=f"CVaR = {cvar:.2f} MW")
    ax.set_xlabel("Scenario loss: duration-weighted incremental DNS [MW]")
    ax.set_ylabel("Scenario probability")
    ax.set_title("Scenario losses and CVaR tail")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_path, show=show)


def plot_cvar_empirical_cdf(results: Mapping[str, Any], output_path: Path, *, show: bool = False) -> None:
    frame = _scenario_losses(results)
    risk = results.get("best_risk") or {}
    if frame.empty:
        return
    ordered = frame.sort_values("loss_mw")
    cumulative = ordered["probability"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(ordered["loss_mw"], cumulative, where="post")
    alpha = _finite(risk.get("alpha"))
    var = _finite(risk.get("var"))
    if math.isfinite(alpha):
        ax.axhline(alpha, linestyle="--", label=f"α = {alpha:.2f}")
    if math.isfinite(var):
        ax.axvline(var, linestyle=":", label=f"VaR = {var:.2f} MW")
    ax.set_xlabel("Scenario loss [MW]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical distribution of schedule-level DNS loss")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_path, show=show)


def plot_schedule_comparison(
    deterministic: Mapping[str, Any],
    cvar: Mapping[str, Any],
    output_path: Path,
    *,
    show: bool = False,
) -> None:
    first = schedule_matrix(deterministic)
    second = schedule_matrix(cvar)
    outages = sorted(set(first.index) | set(second.index), key=lambda name: (not name.startswith("line_"), name))
    columns = sorted(set(first.columns) | set(second.columns))
    first = first.reindex(index=outages, columns=columns, fill_value=0)
    second = second.reindex(index=outages, columns=columns, fill_value=0)
    difference = second - first
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.42 * len(outages) + 1.5)))
    image = ax.imshow(difference.to_numpy(), aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(outages)))
    ax.set_yticklabels(outages)
    tick_step = max(1, len(columns) // 12)
    ticks = np.arange(0, len(columns), tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([columns[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Outage")
    ax.set_title("CVaR schedule minus deterministic schedule")
    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["Deterministic only", "Same", "CVaR only"])
    ax.grid(False)
    _save(fig, output_path, show=show)


def generate_all_plots(
    results_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    show: bool = False,
    compare_path: str | Path | None = None,
) -> Path:
    """Generate all applicable figures and CSV summaries."""
    path = Path(results_path)
    results = load_results(path)
    directory = Path(output_dir) if output_dir is not None else path.with_suffix("").with_name(path.stem + "_figures")
    directory.mkdir(parents=True, exist_ok=True)

    cluster_summary(results).to_csv(directory / "cluster_summary.csv", index=False)
    candidate_summary(results).to_csv(directory / "candidate_summary.csv", index=False)
    _scenario_losses(results).to_csv(directory / "scenario_losses.csv", index=False)

    plot_schedule(results, directory / "01_schedule_heatmap.png", show=show)
    plot_schedule_gantt(results, directory / "02_schedule_gantt.png", show=show)
    plot_active_outage_count(results, directory / "03_concurrent_outages.png", show=show)
    plot_dns_by_cluster(results, directory / "04_dns_by_cluster.png", show=show)
    plot_cluster_severity_timeline(results, directory / "05_cluster_severity_timeline.png", show=show)
    plot_critical_clusters(results, directory / "06_critical_clusters.png", show=show)
    plot_outage_interaction_matrix(results, directory / "07_outage_interaction_matrix.png", show=show)
    plot_worst_contingencies(results, directory / "08_critical_contingencies.png", show=show)
    plot_optimization_convergence(results, directory / "09_optimization_convergence.png", show=show)
    plot_cvar_distribution(results, directory / "10_cvar_scenario_distribution.png", show=show)
    plot_cvar_empirical_cdf(results, directory / "11_cvar_empirical_cdf.png", show=show)

    if compare_path is not None:
        comparison = load_results(compare_path)
        plot_schedule_comparison(results, comparison, directory / "12_schedule_comparison.png", show=show)

    return directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Path to deterministic or CVaR result JSON.")
    parser.add_argument("--output-dir", default=None, help="Directory in which figures and CSV files are written.")
    parser.add_argument("--show", action="store_true", help="Display each figure interactively in addition to saving it.")
    parser.add_argument("--compare", default=None, help="Optional second JSON file for schedule comparison.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    directory = generate_all_plots(
        arguments.results,
        output_dir=arguments.output_dir,
        show=arguments.show,
        compare_path=arguments.compare,
    )
    print(f"Clustered-scheduler figures written to: {directory.resolve()}")


if __name__ == "__main__":
    main()
