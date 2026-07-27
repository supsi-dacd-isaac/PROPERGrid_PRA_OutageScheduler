"""Post-processing and visualization for clustered outage-scheduler results.

This module supports both deterministic and finite-sample CVaR JSON outputs.
It depends only on matplotlib, NumPy, and pandas; no Gurobi model is required.

Examples
--------
python -m scheduler_clustered.results_visualization clustered_deterministic_results.json
python -m scheduler_clustered.results_visualization clustered_cvar_results.json --show
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 180


def step_number(value: Any) -> int:
    """Convert ``step_123`` or a numeric value to an integer index."""
    text = str(value)
    if text.startswith("step_"):
        text = text.split("_", 1)[1]
    return int(float(text))


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_results(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with result_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError("The result JSON root must be an object.")
    return data


def _candidate_signature(candidate: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    starts = candidate.get("start_times") or {}
    return tuple(sorted((str(key), str(value)) for key, value in starts.items()))


def best_candidate(results: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [item for item in results.get("candidates", []) if isinstance(item, Mapping)]
    if not candidates:
        return {}

    schedule = results.get("best_schedule")
    if isinstance(schedule, Mapping):
        signature = tuple(sorted((str(key), str(value)) for key, value in schedule.items()))
        matches = [candidate for candidate in candidates if _candidate_signature(candidate) == signature]
        if matches:
            return max(matches, key=lambda candidate: finite(candidate.get("score"), -math.inf))

    best_score = finite(results.get("best_score"))
    if math.isfinite(best_score):
        return min(
            candidates,
            key=lambda candidate: abs(finite(candidate.get("score"), -math.inf) - best_score),
        )
    return max(candidates, key=lambda candidate: finite(candidate.get("score"), -math.inf))


def candidate_clusters(results: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("full_validation_clusters", "full_validation"):
        records = results.get(key)
        if isinstance(records, list) and records:
            return [item for item in records if isinstance(item, Mapping)]
    candidate = best_candidate(results)
    records = candidate.get("clusters") if isinstance(candidate, Mapping) else None
    return [item for item in (records or []) if isinstance(item, Mapping)]


def ordered_times(results: Mapping[str, Any]) -> list[str]:
    active = results.get("best_active_outages")
    if isinstance(active, Mapping) and active:
        return sorted((str(time) for time in active), key=step_number)

    max_index = -1
    for record in candidate_clusters(results):
        cluster = record.get("cluster") or {}
        max_index = max(max_index, int(cluster.get("end_index", 0)) - 1)
    if max_index < 0:
        schedule = results.get("best_schedule") or {}
        max_index = max((step_number(value) for value in schedule.values()), default=0)
    return [f"step_{index}" for index in range(max_index + 1)]


def schedule_matrix(results: Mapping[str, Any]) -> pd.DataFrame:
    times = ordered_times(results)
    active = results.get("best_active_outages")
    outages: set[str] = set((results.get("best_schedule") or {}).keys())

    if isinstance(active, Mapping):
        for values in active.values():
            outages.update(str(value) for value in (values or []))

    if not outages:
        for record in candidate_clusters(results):
            cluster = record.get("cluster") or {}
            outages.update(str(value) for value in cluster.get("active_outages", []))

    ordered_outages = sorted(outages, key=lambda name: (not name.startswith("line_"), name))
    matrix = pd.DataFrame(0, index=ordered_outages, columns=[step_number(time) for time in times], dtype=int)

    if isinstance(active, Mapping) and active:
        for time, values in active.items():
            index = step_number(time)
            if index not in matrix.columns:
                continue
            for outage in values or []:
                outage_name = str(outage)
                if outage_name in matrix.index:
                    matrix.loc[outage_name, index] = 1
        return matrix

    for record in candidate_clusters(results):
        cluster = record.get("cluster") or {}
        start = int(cluster.get("start_index", 0))
        end = int(cluster.get("end_index", start))
        for outage in cluster.get("active_outages", []):
            outage_name = str(outage)
            if outage_name in matrix.index:
                valid_columns = [column for column in range(start, end) if column in matrix.columns]
                matrix.loc[outage_name, valid_columns] = 1
    return matrix


def cluster_summary(results: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in candidate_clusters(results):
        cluster = record.get("cluster") or {}
        representative = record.get("representative") or {}
        candidate = record.get("candidate") or {}
        baseline = record.get("baseline") or {}
        risk = record.get("risk") or {}
        contingencies = candidate.get("contingencies") or []
        total_incremental = finite(record.get("total_incremental_dns"), 0.0)
        n_states = max(1, len(contingencies))
        expected_incremental = finite(risk.get("expected_loss"), total_incremental / n_states)
        active_outages = [str(value) for value in cluster.get("active_outages", [])]
        rows.append(
            {
                "cluster_id": str(cluster.get("cluster_id", f"cluster_{len(rows)+1:03d}")),
                "start_index": int(cluster.get("start_index", 0)),
                "end_index": int(cluster.get("end_index", 0)),
                "duration_steps": int(cluster.get("duration_steps", 0)),
                "representative_time": representative.get("time"),
                "representative_total_demand_mw": finite(representative.get("total_demand")),
                "active_outages": " + ".join(active_outages),
                "outage_count": len(active_outages),
                "candidate_max_dns_mw": finite(candidate.get("maximum_dns"), 0.0),
                "baseline_max_dns_mw": finite(baseline.get("maximum_dns"), 0.0),
                "incremental_max_dns_mw": finite(record.get("maximum_incremental_dns"), 0.0),
                "incremental_total_dns_mw": total_incremental,
                "expected_incremental_dns_mw": expected_incremental,
                "var_incremental_dns_mw": finite(risk.get("var")),
                "cvar_incremental_dns_mw": finite(risk.get("cvar")),
                "normalized_severity": finite(record.get("normalized_severity"), 0.0),
                "worst_contingency": candidate.get("worst_contingency"),
                "budget_feasible": record.get("budget_feasible"),
            }
        )
    return pd.DataFrame(rows)


def candidate_summary(results: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(results.get("candidates", []), start=1):
        if not isinstance(candidate, Mapping):
            continue
        risk = candidate.get("risk") or {}
        rows.append(
            {
                "iteration": int(candidate.get("iteration", position)),
                "score": finite(candidate.get("score")),
                "maintenance_utility": finite(candidate.get("maintenance_utility")),
                "deterministic_security_cost": finite(candidate.get("deterministic_security_cost")),
                "expected_incremental_dns_mw": finite(risk.get("expected_loss")),
                "var_incremental_dns_mw": finite(risk.get("var")),
                "cvar_incremental_dns_mw": finite(risk.get("cvar")),
                "deferred_outages": len(candidate.get("deferred_outages") or []),
                "cluster_count": len(candidate.get("clusters") or []),
                "budget_feasible": candidate.get("budget_feasible"),
                "risk_feasible": candidate.get("risk_feasible"),
            }
        )
    return pd.DataFrame(rows)


def scenario_loss_summary(results: Mapping[str, Any]) -> pd.DataFrame:
    losses = results.get("best_scenario_losses") or {}
    probabilities = results.get("best_scenario_probabilities") or {}
    risk = results.get("best_risk") or {}
    tail_ids = set(str(value) for value in risk.get("tail_ids", []))
    rows = []
    for scenario_id, loss in losses.items():
        rows.append(
            {
                "scenario_id": str(scenario_id),
                "loss_mw": finite(loss, 0.0),
                "probability": finite(probabilities.get(scenario_id), 0.0),
                "in_cvar_tail": str(scenario_id) in tail_ids,
            }
        )
    return pd.DataFrame(rows).sort_values("loss_mw") if rows else pd.DataFrame()


def contingency_frequency(results: Mapping[str, Any]) -> pd.Series:
    counts: Counter[str] = Counter()
    for record in candidate_clusters(results):
        candidate = record.get("candidate") or {}
        worst = candidate.get("worst_contingency")
        if worst:
            counts[str(worst)] += 1
        for scenario in record.get("scenario_results", []) or []:
            if isinstance(scenario, Mapping) and scenario.get("worst_contingency"):
                counts[str(scenario["worst_contingency"])] += 1
    return pd.Series(dict(counts), dtype=float).sort_values(ascending=False)


def pair_severity_matrix(results: Mapping[str, Any]) -> pd.DataFrame:
    frame = cluster_summary(results)
    names: set[str] = set()
    observations: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in frame.itertuples():
        outages = [part.strip() for part in str(row.active_outages).split("+") if part.strip()]
        names.update(outages)
        if len(outages) == 1:
            observations[(outages[0], outages[0])].append(float(row.normalized_severity))
        for first_index in range(len(outages)):
            for second_index in range(first_index + 1, len(outages)):
                first, second = sorted((outages[first_index], outages[second_index]))
                observations[(first, second)].append(float(row.normalized_severity))
    ordered = sorted(names, key=lambda name: (not name.startswith("line_"), name))
    matrix = pd.DataFrame(0.0, index=ordered, columns=ordered)
    for (first, second), values in observations.items():
        value = float(np.mean(values))
        matrix.loc[first, second] = value
        matrix.loc[second, first] = value
    return matrix


def _save(fig: plt.Figure, path: Path, *, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_schedule_heatmap(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    matrix = schedule_matrix(results)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.45 * len(matrix) + 1.5)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    tick_positions = np.linspace(0, max(0, len(matrix.columns) - 1), min(13, len(matrix.columns)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([matrix.columns[index] for index in tick_positions])
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Planned outage")
    ax.set_title("Optimized outage schedule")
    fig.colorbar(image, ax=ax, label="Outage active")
    _save(fig, path, show=show)


def plot_schedule_gantt(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    matrix = schedule_matrix(results)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.45 * len(matrix) + 1.5)))
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
    ax.set_title("Optimized outage schedule — Gantt view")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path, show=show)


def plot_concurrent_outages(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    matrix = schedule_matrix(results)
    if matrix.empty:
        return
    count = matrix.sum(axis=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(count.index, count.values, where="post")
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Concurrent outages")
    ax.set_title("Maintenance concurrency")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_dns_by_cluster(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    frame = cluster_summary(results)
    if frame.empty:
        return
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(max(12, 0.9 * len(frame)), 5.5))
    ax.plot(x, frame["incremental_max_dns_mw"], marker="o", label="Maximum incremental DNS")
    ax.plot(x, frame["expected_incremental_dns_mw"], marker="s", label="Mean/expected incremental DNS")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["cluster_id"], rotation=45, ha="right")
    ax.set_xlabel("Outage cluster")
    ax.set_ylabel("Incremental DNS [MW]")
    ax.set_title("Incremental DNS by outage cluster")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_cluster_timeline(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    frame = cluster_summary(results)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    for row in frame.itertuples():
        ax.bar(float(row.start_index), float(row.incremental_max_dns_mw), width=float(row.duration_steps), align="edge")
        if float(row.incremental_max_dns_mw) > 0:
            ax.text(
                float(row.start_index) + float(row.duration_steps) / 2,
                float(row.incremental_max_dns_mw),
                str(row.cluster_id),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Maximum incremental DNS [MW]")
    ax.set_title("Cluster security severity over time")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path, show=show)


def plot_critical_clusters(results: Mapping[str, Any], path: Path, *, show: bool = False, top_n: int = 12) -> None:
    frame = cluster_summary(results)
    if frame.empty:
        return
    ranked = frame.sort_values(["normalized_severity", "incremental_max_dns_mw"], ascending=False).head(top_n)
    labels = [f"{row.cluster_id}: {row.active_outages}" for row in ranked.itertuples()]
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.55 * len(ranked) + 1.5)))
    positions = np.arange(len(ranked))
    ax.barh(positions, ranked["normalized_severity"].to_numpy(dtype=float))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Duration-weighted normalized severity")
    ax.set_title("Most critical outage clusters")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path, show=show)


def plot_outage_interactions(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    matrix = pair_severity_matrix(results)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.58 * len(matrix) + 3), max(7, 0.58 * len(matrix) + 2)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Mean severity of outage interactions")
    fig.colorbar(image, ax=ax, label="Normalized severity")
    _save(fig, path, show=show)


def plot_critical_contingencies(results: Mapping[str, Any], path: Path, *, show: bool = False, top_n: int = 15) -> None:
    counts = contingency_frequency(results).head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.43 * len(counts) + 1.5)))
    positions = np.arange(len(counts))
    ax.barh(positions, counts.values)
    ax.set_yticks(positions)
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of worst-state occurrences")
    ax.set_title("Critical N−1 contingencies")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path, show=show)


def plot_objective_convergence(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    frame = candidate_summary(results)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["iteration"], frame["score"], marker="o")
    ax.set_xlabel("Outer-loop iteration")
    ax.set_ylabel("Objective score")
    ax.set_title("Optimization convergence")
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_cvar_convergence(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    frame = candidate_summary(results).dropna(subset=["cvar_incremental_dns_mw"])
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["iteration"], frame["expected_incremental_dns_mw"], marker="o", label="Expected loss")
    ax.plot(frame["iteration"], frame["var_incremental_dns_mw"], marker="s", label="VaR")
    ax.plot(frame["iteration"], frame["cvar_incremental_dns_mw"], marker="^", label="CVaR")
    ax.set_xlabel("Outer-loop iteration")
    ax.set_ylabel("Incremental DNS [MW]")
    ax.set_title("Risk-metric convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_cvar_distribution(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    frame = scenario_loss_summary(results)
    if frame.empty:
        return
    risk = results.get("best_risk") or {}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(frame["loss_mw"].to_numpy(dtype=float), bins=min(15, max(5, len(frame) // 2)))
    var = finite(risk.get("var"))
    cvar = finite(risk.get("cvar"))
    if math.isfinite(var):
        ax.axvline(var, linestyle="--", label=f"VaR = {var:.3g} MW")
    if math.isfinite(cvar):
        ax.axvline(cvar, linestyle=":", label=f"CVaR = {cvar:.3g} MW")
    ax.set_xlabel("Duration-weighted incremental DNS [MW]")
    ax.set_ylabel("Scenario count")
    ax.set_title("Finite-sample loss distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path, show=show)


def plot_cvar_ecdf(results: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    frame = scenario_loss_summary(results)
    if frame.empty:
        return
    values = np.sort(frame["loss_mw"].to_numpy(dtype=float))
    probabilities = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(values, probabilities, where="post")
    ax.set_xlabel("Duration-weighted incremental DNS [MW]")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Empirical CDF of scenario losses")
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def generate_all_plots(
    results_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> Path:
    results_file = Path(results_path)
    results = load_results(results_file)
    directory = Path(output_dir) if output_dir is not None else results_file.with_name(f"{results_file.stem}_visuals")
    directory.mkdir(parents=True, exist_ok=True)

    cluster_summary(results).to_csv(directory / "cluster_summary.csv", index=False)
    candidate_summary(results).to_csv(directory / "candidate_summary.csv", index=False)
    scenario_loss_summary(results).to_csv(directory / "scenario_losses.csv", index=False)

    plot_schedule_heatmap(results, directory / "01_schedule_heatmap.png", show=show)
    plot_schedule_gantt(results, directory / "02_schedule_gantt.png", show=show)
    plot_concurrent_outages(results, directory / "03_concurrent_outages.png", show=show)
    plot_dns_by_cluster(results, directory / "04_dns_by_cluster.png", show=show)
    plot_cluster_timeline(results, directory / "05_cluster_severity_timeline.png", show=show)
    plot_critical_clusters(results, directory / "06_critical_clusters.png", show=show)
    plot_outage_interactions(results, directory / "07_outage_interactions.png", show=show)
    plot_critical_contingencies(results, directory / "08_critical_contingencies.png", show=show)
    plot_objective_convergence(results, directory / "09_objective_convergence.png", show=show)
    plot_cvar_convergence(results, directory / "10_risk_convergence.png", show=show)
    plot_cvar_distribution(results, directory / "11_cvar_loss_distribution.png", show=show)
    plot_cvar_ecdf(results, directory / "12_cvar_loss_ecdf.png", show=show)
    return directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Deterministic or CVaR result JSON.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    directory = generate_all_plots(arguments.results, output_dir=arguments.output_dir, show=arguments.show)
    print(f"Clustered-scheduler figures written to: {directory.resolve()}")


if __name__ == "__main__":
    main()
