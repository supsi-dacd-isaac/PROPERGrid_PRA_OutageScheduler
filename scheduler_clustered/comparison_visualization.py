"""Deterministic-versus-CVaR comparison visualizations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scheduler_clustered.results_visualization import (
    FIGURE_DPI,
    candidate_summary,
    cluster_summary,
    contingency_frequency,
    finite,
    load_results,
    pair_severity_matrix,
    schedule_matrix,
    step_number,
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _save(fig: plt.Figure, path: Path, *, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def start_comparison(det: Mapping[str, Any], cvar: Mapping[str, Any]) -> pd.DataFrame:
    det_schedule = det.get("best_schedule") or {}
    cvar_schedule = cvar.get("best_schedule") or {}
    outages = sorted(set(det_schedule) | set(cvar_schedule), key=lambda name: (not str(name).startswith("line_"), str(name)))
    rows = []
    for outage in outages:
        det_label = det_schedule.get(outage)
        cvar_label = cvar_schedule.get(outage)
        det_index = step_number(det_label) if det_label is not None else math.nan
        cvar_index = step_number(cvar_label) if cvar_label is not None else math.nan
        shift = cvar_index - det_index if math.isfinite(det_index) and math.isfinite(cvar_index) else math.nan
        rows.append(
            {
                "outage": str(outage),
                "deterministic_start": det_label,
                "cvar_start": cvar_label,
                "deterministic_start_index": det_index,
                "cvar_start_index": cvar_index,
                "cvar_minus_deterministic_steps": shift,
                "same_start": bool(math.isfinite(shift) and shift == 0),
            }
        )
    return pd.DataFrame(rows)


def formulation_summary(det: Mapping[str, Any], cvar: Mapping[str, Any]) -> pd.DataFrame:
    risk = cvar.get("best_risk") or {}
    return pd.DataFrame(
        [
            {
                "formulation": "deterministic",
                "termination_reason": det.get("termination_reason"),
                "best_score": finite(det.get("best_score")),
                "deterministic_security_cost": finite(det.get("deterministic_security_cost")),
                "expected_incremental_dns_mw": math.nan,
                "var_incremental_dns_mw": math.nan,
                "cvar_incremental_dns_mw": math.nan,
                "deferred_outages": len(det.get("deferred_outages") or []),
                "candidate_count": len(det.get("candidates") or []),
            },
            {
                "formulation": "cvar",
                "termination_reason": cvar.get("termination_reason"),
                "best_score": finite(cvar.get("best_score")),
                "deterministic_security_cost": math.nan,
                "expected_incremental_dns_mw": finite(risk.get("expected_loss")),
                "var_incremental_dns_mw": finite(risk.get("var")),
                "cvar_incremental_dns_mw": finite(risk.get("cvar")),
                "deferred_outages": len(cvar.get("deferred_outages") or []),
                "candidate_count": len(cvar.get("candidates") or []),
            },
        ]
    )


def _aligned_schedule_matrices(det: Mapping[str, Any], cvar: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    det_matrix = schedule_matrix(det)
    cvar_matrix = schedule_matrix(cvar)
    outages = sorted(set(det_matrix.index) | set(cvar_matrix.index), key=lambda name: (not str(name).startswith("line_"), str(name)))
    columns = sorted(set(det_matrix.columns) | set(cvar_matrix.columns))
    return (
        det_matrix.reindex(index=outages, columns=columns, fill_value=0),
        cvar_matrix.reindex(index=outages, columns=columns, fill_value=0),
    )


def plot_schedule_difference(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    det_matrix, cvar_matrix = _aligned_schedule_matrices(det, cvar)
    if det_matrix.empty:
        return
    difference = cvar_matrix.to_numpy(dtype=int) - det_matrix.to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.45 * len(det_matrix) + 1.5)))
    image = ax.imshow(difference, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(det_matrix.index)))
    ax.set_yticklabels(det_matrix.index)
    positions = np.linspace(0, max(0, len(det_matrix.columns) - 1), min(13, len(det_matrix.columns)), dtype=int)
    ax.set_xticks(positions)
    ax.set_xticklabels([det_matrix.columns[index] for index in positions])
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Planned outage")
    ax.set_title("Schedule difference: CVaR minus deterministic")
    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["Deterministic only", "Same", "CVaR only"])
    _save(fig, path, show=show)


def plot_start_shifts(starts: pd.DataFrame, path: Path, *, show: bool = False) -> None:
    valid = starts.dropna(subset=["cvar_minus_deterministic_steps"]).sort_values("cvar_minus_deterministic_steps")
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42 * len(valid) + 1.5)))
    ax.barh(valid["outage"], valid["cvar_minus_deterministic_steps"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("CVaR start minus deterministic start [steps]")
    ax.set_ylabel("Outage")
    ax.set_title("Outage-start shifts induced by CVaR")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path, show=show)


def plot_concurrency_comparison(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    det_matrix, cvar_matrix = _aligned_schedule_matrices(det, cvar)
    if det_matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.step(det_matrix.columns, det_matrix.sum(axis=0), where="post", label="Deterministic")
    ax.step(cvar_matrix.columns, cvar_matrix.sum(axis=0), where="post", label="CVaR")
    ax.set_xlabel("Planning step")
    ax.set_ylabel("Concurrent outages")
    ax.set_title("Maintenance concurrency comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def _normalized_best_so_far(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["score"]).sort_values("iteration").copy()
    if valid.empty:
        return valid
    best = valid["score"].cummax()
    initial = float(best.iloc[0])
    scale = max(abs(initial), 1e-12)
    valid["relative_best_score_percent"] = 100.0 * (best - initial) / scale
    return valid


def plot_search_progress(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    det_frame = _normalized_best_so_far(candidate_summary(det))
    cvar_frame = _normalized_best_so_far(candidate_summary(cvar))
    if det_frame.empty and cvar_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if not det_frame.empty:
        ax.plot(det_frame["iteration"], det_frame["relative_best_score_percent"], marker="o", label="Deterministic")
    if not cvar_frame.empty:
        ax.plot(cvar_frame["iteration"], cvar_frame["relative_best_score_percent"], marker="s", label="CVaR")
    ax.set_xlabel("Outer-loop iteration")
    ax.set_ylabel("Best-score improvement relative to first candidate [%]")
    ax.set_title("Within-formulation search progress")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def plot_cluster_dns_ecdf(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    det_values = cluster_summary(det)["incremental_max_dns_mw"].dropna().to_numpy(dtype=float)
    cvar_values = cluster_summary(cvar)["incremental_max_dns_mw"].dropna().to_numpy(dtype=float)
    if len(det_values) == 0 and len(cvar_values) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(det_values):
        x, y = _ecdf(det_values)
        ax.step(x, y, where="post", label="Deterministic")
    if len(cvar_values):
        x, y = _ecdf(cvar_values)
        ax.step(x, y, where="post", label="CVaR")
    ax.set_xlabel("Maximum incremental DNS per cluster [MW]")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Cluster-severity distribution comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def plot_ranked_critical_clusters(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False, top_n: int = 8) -> None:
    rows = []
    for label, results in (("DET", det), ("CVaR", cvar)):
        frame = cluster_summary(results).sort_values("incremental_max_dns_mw", ascending=False).head(top_n)
        for row in frame.itertuples():
            rows.append({"label": f"{label} {row.cluster_id}: {row.active_outages}", "value": float(row.incremental_max_dns_mw)})
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("value")
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.42 * len(frame) + 1.5)))
    ax.barh(frame["label"], frame["value"])
    ax.set_xlabel("Maximum incremental DNS [MW]")
    ax.set_title("Most critical clusters in each formulation")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, path, show=show)


def plot_contingency_frequency_comparison(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False, top_n: int = 15) -> None:
    det_counts = contingency_frequency(det)
    cvar_counts = contingency_frequency(cvar)
    combined = pd.concat([det_counts.rename("deterministic"), cvar_counts.rename("cvar")], axis=1).fillna(0.0)
    if combined.empty:
        return
    combined["total"] = combined.sum(axis=1)
    combined = combined.sort_values("total", ascending=False).head(top_n).drop(columns="total")
    positions = np.arange(len(combined))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(combined) + 4), 5.5))
    ax.bar(positions - width / 2, combined["deterministic"], width, label="Deterministic")
    ax.bar(positions + width / 2, combined["cvar"], width, label="CVaR")
    ax.set_xticks(positions)
    ax.set_xticklabels(combined.index, rotation=45, ha="right")
    ax.set_xlabel("Contingency")
    ax.set_ylabel("Worst-state occurrence count")
    ax.set_title("Critical-contingency comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path, show=show)


def plot_interaction_difference(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    det_matrix = pair_severity_matrix(det)
    cvar_matrix = pair_severity_matrix(cvar)
    names = sorted(set(det_matrix.index) | set(cvar_matrix.index), key=lambda name: (not str(name).startswith("line_"), str(name)))
    if not names:
        return
    difference = cvar_matrix.reindex(index=names, columns=names, fill_value=0.0) - det_matrix.reindex(index=names, columns=names, fill_value=0.0)
    limit = float(np.nanmax(np.abs(difference.to_numpy()))) if difference.size else 0.0
    limit = max(limit, 1e-12)
    fig, ax = plt.subplots(figsize=(max(8, 0.58 * len(names) + 3), max(7, 0.58 * len(names) + 2)))
    image = ax.imshow(difference.to_numpy(), aspect="auto", interpolation="nearest", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Outage-interaction severity difference: CVaR minus deterministic")
    fig.colorbar(image, ax=ax, label="Difference in normalized severity")
    _save(fig, path, show=show)


def plot_duration_severity(det: Mapping[str, Any], cvar: Mapping[str, Any], path: Path, *, show: bool = False) -> None:
    det_frame = cluster_summary(det)
    cvar_frame = cluster_summary(cvar)
    if det_frame.empty and cvar_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    if not det_frame.empty:
        ax.scatter(det_frame["duration_steps"], det_frame["incremental_max_dns_mw"], label="Deterministic")
    if not cvar_frame.empty:
        ax.scatter(cvar_frame["duration_steps"], cvar_frame["incremental_max_dns_mw"], label="CVaR")
    ax.set_xlabel("Cluster duration [steps]")
    ax.set_ylabel("Maximum incremental DNS [MW]")
    ax.set_title("Cluster duration versus security severity")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, show=show)


def generate_comparison_visuals(
    deterministic_path: str | Path,
    cvar_path: str | Path,
    *,
    output_dir: str | Path,
    show: bool = False,
) -> Path:
    det = load_results(deterministic_path)
    cvar = load_results(cvar_path)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    starts = start_comparison(det, cvar)
    summary = formulation_summary(det, cvar)
    shifts = starts["cvar_minus_deterministic_steps"].dropna() if not starts.empty else pd.Series(dtype=float)
    schedule_statistics = {
        "number_of_outages": int(len(starts)),
        "same_start_count": int(starts["same_start"].sum()) if not starts.empty else 0,
        "shifted_outage_count": int((~starts["same_start"]).sum()) if not starts.empty else 0,
        "mean_absolute_start_shift_steps": float(shifts.abs().mean()) if not shifts.empty else math.nan,
        "maximum_absolute_start_shift_steps": float(shifts.abs().max()) if not shifts.empty else math.nan,
    }

    starts.to_csv(directory / "outage_start_comparison.csv", index=False)
    summary.to_csv(directory / "formulation_summary.csv", index=False)
    cluster_summary(det).assign(formulation="deterministic").to_csv(directory / "deterministic_cluster_summary.csv", index=False)
    cluster_summary(cvar).assign(formulation="cvar").to_csv(directory / "cvar_cluster_summary.csv", index=False)
    (directory / "comparison_summary.json").write_text(
        json.dumps(
            json_safe({"formulations": summary.to_dict(orient="records"), "schedule_statistics": schedule_statistics}),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    plot_schedule_difference(det, cvar, directory / "01_schedule_difference.png", show=show)
    plot_start_shifts(starts, directory / "02_outage_start_shifts.png", show=show)
    plot_concurrency_comparison(det, cvar, directory / "03_concurrency_comparison.png", show=show)
    plot_search_progress(det, cvar, directory / "04_search_progress.png", show=show)
    plot_cluster_dns_ecdf(det, cvar, directory / "05_cluster_dns_ecdf.png", show=show)
    plot_ranked_critical_clusters(det, cvar, directory / "06_critical_clusters_comparison.png", show=show)
    plot_contingency_frequency_comparison(det, cvar, directory / "07_contingency_frequency_comparison.png", show=show)
    plot_interaction_difference(det, cvar, directory / "08_outage_interaction_difference.png", show=show)
    plot_duration_severity(det, cvar, directory / "09_duration_severity_comparison.png", show=show)
    return directory
