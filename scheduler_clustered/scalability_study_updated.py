"""Benchmark clustered scheduling versus grid, outage-list and horizon size.

This updated runner can execute any subset of the three scalability axes and
exposes the CVaR exploration controls needed to test diverse schedules. It
keeps solver budgets fixed, executes points sequentially, saves every run, and
writes CSV/JSON/Markdown/LaTeX/PNG evidence plus a guarded reviewer response.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_DIR, PROJECT_ROOT = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkPoint:
    axis: str
    case: str
    config: Path
    requested_value: int
    horizon_steps: int | None = None
    outage_count: int | None = None

    @property
    def label(self) -> str:
        suffix = {"grid": self.case, "outages": f"{self.case}_o{self.requested_value}", "horizon": f"{self.case}_h{self.requested_value}"}[self.axis]
        return _safe_label(f"{self.axis}_{suffix}")


def _labelled_path(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.") from exc
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("LABEL and PATH must be non-empty.")
    return label.strip(), Path(path.strip())


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--grid-case", "--case", action="append", type=_labelled_path, required=True, metavar="LABEL=CONFIG", help="Repeat for IEEE24 and IEEE118.")
    add("--reference-case", help="Case used for outage-list and horizon sweeps; defaults to the first grid case.")
    add("--axes", nargs="+", choices=("grid", "outages", "horizon"), default=("grid", "outages", "horizon"), help="Run only the selected scalability axes.")
    add("--outage-counts", nargs="+", type=int, help="Planned-outage counts; default: three automatic scales.")
    add("--horizons", nargs="+", type=int, help="Horizon lengths; default: three automatic scales.")
    add("--repeats", type=int, default=1)
    add("--profile", choices=("evidence", "fast"), default="evidence")
    add("--project-root", type=Path, default=PROJECT_ROOT)
    add("--deterministic-config", type=Path, default=PACKAGE_DIR / "config_clustered_example.json")
    add("--cvar-config", type=Path, default=PACKAGE_DIR / "config_clustered_cvar_example.json")
    add("--output-root", type=Path, default=Path("outputs/wp3_scalability"))
    add("--validation-samples", type=int, default=0, help="0 times optimisation only; use >0 for paired validation at every point.")
    add("--validation-workers", type=int, default=1)
    add("--validation-seed", type=int, default=20260727)
    add("--validation-alpha", type=float, default=0.95)
    add("--oracle-workers", type=int)
    add("--master-threads", type=int)
    add("--contingency-top-k", type=int)
    add("--scenario-count", type=int)
    add("--max-candidates", type=int)
    add("--patience", type=int, help="Stop after this many consecutive candidates without improvement.")
    add("--min-candidate-changes", type=int, help="Minimum number of outage decisions that every new CVaR candidate must change.")
    add("--relative-utility-tolerance", type=float, help="Relative deterministic-utility loss allowed for CVaR selection.")
    add("--risk-proxy-learning-rate", type=float)
    add("--scenario-seed", type=int)
    add("--cvar-alpha", type=float)
    add("--critical-states", type=int)
    add("--master-time-limit", type=float)
    add("--master-mip-gap", type=float)
    add("--reuse-point", action="append", type=_labelled_path, default=[], metavar="POINT=RUN_DIR")
    add("--continue-on-error", action="store_true")
    add("--dry-run", action="store_true")
    add("--echo", action="store_true")
    return parser


def _base_dimensions(config: Path, clustered_config: Path) -> dict[str, Any]:
    from .data_adapter import power_data_diagnostics
    from .runtime import apply_data_overrides_from_json, configured_decomposition_data

    data = configured_decomposition_data(config, allow_outage_deferral=True)
    apply_data_overrides_from_json(data, clustered_config)
    diagnostics = power_data_diagnostics(data)
    return {
        "buses": int(diagnostics["buses"]),
        "branches": int(diagnostics["branches"]),
        "generators": int(diagnostics["generators_represented"]),
        "outages": len(data["names"]["outages"]),
        "horizon": len(data["T"]),
        "max_duration": max(int(round(float(value))) for value in data["durations"].values()),
    }


def _auto_scales(maximum: int, minimum: int = 1) -> list[int]:
    values = {maximum, max(minimum, int(round(maximum / 2))), max(minimum, int(round(maximum / 4)))}
    return sorted(value for value in values if minimum <= value <= maximum)


def _points(
    cases: list[tuple[str, Path]],
    reference: str,
    case_dimensions: Mapping[str, Mapping[str, Any]],
    outages: Sequence[int] | None,
    horizons: Sequence[int] | None,
    axes: Sequence[str],
) -> list[BenchmarkPoint]:
    dimensions = case_dimensions[reference]
    reference_config = dict(cases)[reference]
    selected = set(axes)
    points: list[BenchmarkPoint] = []
    if "grid" in selected:
        points += [BenchmarkPoint("grid", label, config, int(case_dimensions[label]["buses"])) for label, config in cases]
    if "outages" in selected:
        values = sorted(set(outages or _auto_scales(int(dimensions["outages"]))))
        points += [BenchmarkPoint("outages", reference, reference_config, value, outage_count=value) for value in values]
    if "horizon" in selected:
        values = sorted(set(horizons or _auto_scales(int(dimensions["horizon"]), int(dimensions["max_duration"]))))
        points += [BenchmarkPoint("horizon", reference, reference_config, value, horizon_steps=value) for value in values]
    return points


def _setting_overrides(args: argparse.Namespace, section: str) -> dict[str, Any]:
    values = {
        "oracle_workers": args.oracle_workers,
        "master_threads": args.master_threads,
        "contingency_top_k": args.contingency_top_k,
        "max_candidates": args.max_candidates,
        "critical_state_count": args.critical_states,
        "master_time_limit": args.master_time_limit,
        "master_mip_gap": args.master_mip_gap,
    }
    if section == "clustered_cvar":
        values.update({
            "scenario_count": args.scenario_count,
            "patience": args.patience,
            "min_candidate_changes": args.min_candidate_changes,
            "relative_utility_tolerance": args.relative_utility_tolerance,
            "risk_proxy_learning_rate": args.risk_proxy_learning_rate,
            "scenario_seed": args.scenario_seed,
            "cvar_alpha": args.cvar_alpha,
        })
    if args.profile == "fast":
        fast = {"oracle_workers": 4, "master_threads": 4, "contingency_top_k": 5, "max_candidates": 8, "critical_state_count": 2, "master_time_limit": 90.0, "master_mip_gap": 0.02}
        if section == "clustered_cvar":
            fast["scenario_count"] = 12
        values = {key: values.get(key) if values.get(key) is not None else value for key, value in fast.items()}
    result = {key: value for key, value in values.items() if value is not None}
    if section == "clustered_cvar" and "scenario_count" in result:
        result["gaussian_samples_per_critical_state"] = None
    return result


def _variant(base: Path, section: str, point: BenchmarkPoint, target: Path, results_path: Path, args: argparse.Namespace) -> Path:
    document = _read(base)
    settings = dict(document.get(section, document))
    settings.update(_setting_overrides(args, section))
    settings["results_path"] = str(results_path)
    document = {**document, section: settings} if section in document else {section: settings}
    if point.horizon_steps is not None or point.outage_count is not None:
        document["benchmark_slice"] = {
            "normalise_windows": True,
            **({"horizon_steps": point.horizon_steps} if point.horizon_steps is not None else {}),
            **({"outage_count": point.outage_count} if point.outage_count is not None else {}),
        }
    _write(target, document)
    return target


def _command(point: BenchmarkPoint, run_dir: Path, deterministic: Path, cvar: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-m", "scheduler_clustered.run_cluster_comparison",
        "--project-root", str(args.project_root), "--config", str(point.config),
        "--deterministic-config", str(deterministic), "--cvar-config", str(cvar),
        "--output-dir", str(run_dir), "--skip-plots",
    ]
    if args.validation_samples <= 0:
        command.append("--skip-validation")
    else:
        command += [
            "--validation-samples", str(args.validation_samples),
            "--validation-workers", str(args.validation_workers),
            "--validation-seed", str(args.validation_seed),
            "--validation-alpha", str(args.validation_alpha),
        ]
    return command


def _run(command: Sequence[str], cwd: Path, log_path: Path, echo: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(list(command), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if echo:
                print(line, end="")
        code = process.wait()
    if code:
        raise RuntimeError(f"Benchmark subprocess failed with exit code {code}; see {log_path}.")
    return float(time.perf_counter() - started)


def _stage(performance: Mapping[str, Any], name: str) -> float | None:
    value = ((performance.get("stages") or {}).get(name) or {}).get("wall_time_seconds")
    return None if value is None else float(value)


def _candidate_diversity(document: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise realised Hamming distances between evaluated schedules."""
    candidates = list(document.get("candidates") or [])

    def decisions(candidate: Mapping[str, Any]) -> dict[str, str]:
        result = {str(key): str(value) for key, value in (candidate.get("start_times") or {}).items()}
        result.update({str(key): "<deferred>" for key in candidate.get("deferred_outages") or ()})
        return result

    schedules = [decisions(candidate) for candidate in candidates]
    distances = [
        sum(left.get(key) != right.get(key) for key in set(left) | set(right))
        for index, left in enumerate(schedules)
        for right in schedules[index + 1:]
    ]
    return {
        "cvar_unique_schedules": len({tuple(sorted(item.items())) for item in schedules}),
        "minimum_realised_candidate_distance": min(distances) if distances else None,
        "median_realised_candidate_distance": _median(distances),
    }


def _row(point: BenchmarkPoint, repeat: int, run_dir: Path, driver_seconds: float | None) -> dict[str, Any]:
    manifest = _read(run_dir / "run_manifest.json")
    performance = manifest.get("performance") or {}
    dimensions = performance.get("problem_dimensions") or {}
    deterministic, cvar, validation = performance.get("deterministic") or {}, performance.get("cvar") or {}, performance.get("validation") or {}
    cvar_result = _read(run_dir / "results" / "clustered_cvar_results.json")
    cvar_config = cvar_result.get("config") or {}
    size = {"grid": dimensions.get("buses"), "outages": dimensions.get("planned_outages"), "horizon": dimensions.get("horizon_steps")}[point.axis]
    return {
        "point": point.label, "axis": point.axis, "case": point.case, "requested_value": point.requested_value, "actual_size": size,
        "repeat": repeat, "status": "completed", "buses": dimensions.get("buses"), "branches": dimensions.get("branches"),
        "generators": dimensions.get("generators"), "planned_outages": dimensions.get("planned_outages"),
        "n_minus_1_contingencies": dimensions.get("n_minus_1_contingencies"), "horizon_steps": dimensions.get("horizon_steps"),
        "selected_outages": ",".join((dimensions.get("benchmark_slice") or {}).get("outages", [])) or None,
        "deterministic_seconds": _stage(performance, "deterministic_optimization"), "cvar_seconds": _stage(performance, "cvar_optimization"),
        "validation_seconds": _stage(performance, "paired_validation"), "total_seconds": performance.get("workflow_wall_time_seconds", driver_seconds),
        "peak_rss_mb": performance.get("child_peak_rss_mb"), "deterministic_candidates": deterministic.get("candidate_count"),
        "deterministic_scopf_solves": (deterministic.get("oracle_statistics") or {}).get("solve_calls"),
        "cvar_scopf_solves": (cvar.get("oracle_statistics") or {}).get("solve_calls"),
        "deterministic_seconds_per_scopf": (deterministic.get("oracle_statistics") or {}).get("mean_seconds_per_solve"),
        "cvar_seconds_per_scopf": (cvar.get("oracle_statistics") or {}).get("mean_seconds_per_solve"),
        "deterministic_candidate_budget": deterministic.get("max_candidates"), "cvar_candidates": cvar.get("candidate_count"),
        "cvar_candidate_budget": cvar.get("max_candidates"), "deterministic_termination": deterministic.get("termination_reason"),
        "cvar_termination": cvar.get("termination_reason"), "benchmark_included": cvar.get("benchmark_included"),
        "cvar_patience": cvar_config.get("patience"), "min_candidate_changes": cvar_config.get("min_candidate_changes", 1),
        **_candidate_diversity(cvar_result),
        "critical_states": cvar.get("critical_state_count"), "scenarios": cvar.get("scenario_count"),
        "contingency_top_k": cvar.get("contingency_top_k"), "master_threads": cvar.get("master_threads"),
        "oracle_workers": cvar.get("oracle_workers"), "master_time_limit": cvar.get("master_time_limit_seconds"),
        "master_mip_gap": cvar.get("master_mip_gap"), "full_contingency_validation": cvar.get("full_contingency_validation"),
        "validation_enabled": validation.get("enabled"), "validation_scenarios": validation.get("scenario_count"),
        "run_directory": str(run_dir.resolve()),
        "software_environment": performance.get("software_environment"), "error": None,
    }


def _median(values: Sequence[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def _group(rows: Sequence[dict[str, Any]], key) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(key(row), []).append(row)
    return result


def _log_slope(points: Sequence[tuple[float, float]]) -> float | None:
    points = [(x, y) for x, y in points if x > 0 and y > 0]
    if len({x for x, _ in points}) < 2:
        return None
    xs, ys = [math.log(x) for x, _ in points], [math.log(y) for _, y in points]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator if denominator else None


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    groups = _group(completed, lambda row: (row["axis"], row["case"], row["actual_size"]))
    summary: list[dict[str, Any]] = []
    for (axis, case, size), records in sorted(groups.items()):
        totals = [row["total_seconds"] for row in records if row.get("total_seconds") is not None]
        item = dict(records[0])
        item.update({
            "repeats_completed": len(records), "median_total_seconds": _median(totals),
            "min_total_seconds": min(totals) if totals else None, "max_total_seconds": max(totals) if totals else None,
            "median_deterministic_seconds": _median([row.get("deterministic_seconds") for row in records]),
            "median_cvar_seconds": _median([row.get("cvar_seconds") for row in records]),
            "median_validation_seconds": _median([row.get("validation_seconds") for row in records]),
        })
        summary.append(item)
    for records in _group(summary, lambda row: (row["axis"], row["case"])).values():
        records.sort(key=lambda row: float(row["actual_size"]))
        baseline = float(records[0]["median_total_seconds"] or math.nan)
        exponent = _log_slope([(float(row["actual_size"]), float(row["median_total_seconds"])) for row in records if row.get("actual_size") and row.get("median_total_seconds")])
        for row in records:
            row.update({
                "runtime_ratio_to_smallest": float(row["median_total_seconds"]) / baseline if baseline > 0 else None,
                "descriptive_log_log_slope": exponent,
            })
    return summary


def _matched_grid_settings(rows: Sequence[dict[str, Any]]) -> bool:
    keys = (
        "horizon_steps", "deterministic_candidate_budget", "cvar_candidate_budget",
        "critical_states", "scenarios", "contingency_top_k", "master_threads",
        "oracle_workers", "master_time_limit", "master_mip_gap",
        "full_contingency_validation", "validation_enabled", "validation_scenarios",
    )
    grid = [row for row in rows if row.get("axis") == "grid" and row.get("status") == "completed"]
    return bool(grid) and all(tuple(row.get(key) for key in keys) == tuple(grid[0].get(key) for key in keys) for row in grid[1:])


def _assessment(rows: list[dict[str, Any]], summary: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    grid = [row for row in summary if row["axis"] == "grid"]
    ieee118 = any(int(row.get("buses") or 0) >= 118 for row in grid)
    axes = {axis: len({row["actual_size"] for row in summary if row["axis"] == axis}) >= 2 for axis in ("grid", "outages", "horizon")}
    matched = _matched_grid_settings(completed)
    benchmark = bool(completed) and all(row.get("benchmark_included") for row in completed)
    bounded = profile == "evidence" and ieee118 and len(grid) >= 2 and matched and benchmark
    interpretation = (
        "The evidence supports only the bounded statement that the clustered implementation completed the reported IEEE-118 benchmark under the stated settings. It does not establish asymptotic or production-grid scalability."
        if bounded else
        "The bounded IEEE-118 claim is not yet supported by these outputs. Do not report scalability until IEEE-118 and a comparison case complete with matched evidence-profile settings and the deterministic benchmark is included in every CVaR candidate pool."
    )
    return {
        "profile": profile, "ieee118_completed": ieee118, "matched_grid_settings": matched,
        "deterministic_benchmark_in_all_cvar_pools": benchmark, "axis_has_at_least_two_sizes": axes,
        "three_axis_study_completed": all(axes.values()), "bounded_ieee118_claim_supported": bounded,
        "asymptotic_scalability_claim_supported": False, "interpretation": interpretation,
    }


_TABLE = (
    ("axis", "Axis"), ("case", "Case"), ("actual_size", "Size"), ("repeats_completed", "Runs"),
    ("median_deterministic_seconds", "Det. [s]"), ("median_cvar_seconds", "CVaR [s]"),
    ("median_validation_seconds", "Validation [s]"), ("median_total_seconds", "Total [s]"),
    ("deterministic_scopf_solves", "Det. SCOPFs"), ("cvar_scopf_solves", "CVaR SCOPFs"),
    ("runtime_ratio_to_smallest", "Ratio"), ("descriptive_log_log_slope", "Log slope"),
    ("minimum_realised_candidate_distance", "Min. distance"),
    ("peak_rss_mb", "RSS [MiB]"), ("deterministic_termination", "Det. stop"), ("cvar_termination", "CVaR stop"),
)


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.4g}"
    return str(value)


def _escape(value: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in _display(value))


def _reviewer_response(summary: list[dict[str, Any]], assessment: Mapping[str, Any]) -> str:
    grid = sorted((row for row in summary if row["axis"] == "grid"), key=lambda row: int(row.get("buses") or 0))
    if assessment["bounded_ieee118_claim_supported"] and len(grid) >= 2:
        cases = "; ".join(
            f"{row['case']} ({int(row.get('buses') or 0)} buses, {int(row.get('n_minus_1_contingencies') or 0)} N-1 states): {row['median_total_seconds']:.1f} s median total runtime"
            for row in grid
        )
        return f"> We agree that the preliminary IEEE-24-only evidence was insufficient. We therefore added matched-budget experiments on the original case and IEEE-118, and complementary sweeps over planned-outage count and horizon length. The measured grid-size results were {cases}. All settings, termination conditions, candidate counts and memory measurements are reported in the accompanying table. We have narrowed the claim to a demonstrated application on IEEE-118; these experiments do not establish asymptotic or production-grid scalability.\n"
    return "> We agree that the preliminary IEEE-24-only evidence was insufficient. A three-axis benchmark and report pipeline has now been added, but the completed outputs do not yet satisfy the IEEE-118 evidence guardrails. We therefore do not claim scalability from the current run. The report will be updated only after IEEE-118 and the comparison case complete with matched evidence-profile settings.\n"


def _plot(summary: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = {"grid": "Buses", "outages": "Planned outages", "horizon": "Horizon steps"}
    for ax, axis in zip(axes, ("grid", "outages", "horizon")):
        records = sorted((row for row in summary if row["axis"] == axis), key=lambda row: float(row["actual_size"]))
        if records:
            ax.plot([row["actual_size"] for row in records], [row["median_total_seconds"] for row in records], "o-", linewidth=2)
        ax.set(xlabel=labels[axis], ylabel="Median wall time [s]", title=axis.capitalize())
        ax.grid(True, alpha=.3)
    fig.suptitle("Clustered scheduler scalability (measured points only)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(rows: list[dict[str, Any]], run_root: Path, profile: str, plan: Sequence[BenchmarkPoint]) -> dict[str, Any]:
    summary = _aggregate(rows)
    assessment = _assessment(rows, summary, profile)
    serial_plan = [{**point.__dict__, "config": str(point.config)} for point in plan]
    evidence = {"benchmark_plan": serial_plan, "raw_runs": rows, "summary": summary, "claim_assessment": assessment}
    _write(run_root / "scalability_results.json", evidence)
    for name, records in (("scalability_runs.csv", rows), ("scalability_summary.csv", summary)):
        fields = sorted({key for row in records for key in row if key != "software_environment"})
        with (run_root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if fields:
                writer.writeheader()
                writer.writerows(records)
    headers = [label for _, label in _TABLE]
    markdown = [
        "# WP3 clustered-scheduler scalability report", "",
        f"Profile: **{profile}**. Only completed measured runs appear below.", "",
        "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(_display(row.get(key)) for key, _ in _TABLE) + " |" for row in summary],
        "", "## Claim assessment", "", assessment["interpretation"], "",
        "The log-log slopes are descriptive fits to the measured points, not complexity proofs.", "",
        "## Suggested response to the reviewer", "", _reviewer_response(summary, assessment),
    ]
    (run_root / "scalability_report.md").write_text("\n".join(markdown), encoding="utf-8")
    (run_root / "reviewer_response.md").write_text(_reviewer_response(summary, assessment), encoding="utf-8")
    latex = [
        r"\begin{table}[t]", r"\centering\scriptsize", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "ll" + "r" * (len(_TABLE) - 4) + "ll}",
        r"\toprule", " & ".join(_escape(value) for value in headers) + r" \\",
        r"\midrule",
        *[" & ".join(_escape(row.get(key)) for key, _ in _TABLE) + r" \\" for row in summary],
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\caption{Measured scalability of the clustered scheduler. Slopes are descriptive only.}",
        r"\label{tab:wp3_scalability}", r"\end{table}",
    ]
    (run_root / "scalability_summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    _plot(summary, run_root / "scalability_runtime.png")
    return evidence


def main_scalability_updated(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.min_candidate_changes is not None and args.min_candidate_changes < 1:
        raise ValueError("--min-candidate-changes must be positive.")
    cases = [(label, _resolve(args.project_root, path)) for label, path in args.grid_case]
    case_names = {label for label, _ in cases}
    reference = args.reference_case or cases[0][0]
    if reference not in case_names:
        raise ValueError(f"Unknown --reference-case {reference!r}.")
    deterministic_base, cvar_base = _resolve(args.project_root, args.deterministic_config), _resolve(args.project_root, args.cvar_config)
    for path in (deterministic_base, cvar_base, *(config for _, config in cases)):
        if not path.is_file():
            raise FileNotFoundError(path)
    case_dimensions = {label: _base_dimensions(config, deterministic_base) for label, config in cases}
    dimensions = case_dimensions[reference]
    plan = _points(cases, reference, case_dimensions, args.outage_counts, args.horizons, args.axes)
    if not plan:
        raise ValueError("No benchmark points were selected.")
    output_root = _resolve(args.project_root, args.output_root)
    run_root = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=False)
    _write(run_root / "benchmark_plan.json", {"profile": args.profile, "case_dimensions": case_dimensions, "reference_dimensions": dimensions, "points": [{**point.__dict__, "config": str(point.config)} for point in plan]})
    if args.dry_run:
        _write_outputs([], run_root, args.profile, plan)
        print(f"Dry-run plan written to: {run_root}")
        return 0
    reuse = {label: _resolve(args.project_root, path) for label, path in args.reuse_point}
    rows: list[dict[str, Any]] = []
    for point in plan:
        for repeat in range(1, args.repeats + 1):
            key = f"{point.label}_r{repeat:02d}"
            run_dir = run_root / "cases" / key
            print(f"\n=== {key} ===")
            try:
                if key in reuse:
                    run_dir, elapsed = reuse[key], None
                else:
                    variant_dir = run_root / "variants"
                    deterministic = _variant(deterministic_base, "clustered_deterministic", point, variant_dir / f"{key}_det.json", run_dir / "raw" / "deterministic.json", args)
                    cvar = _variant(cvar_base, "clustered_cvar", point, variant_dir / f"{key}_cvar.json", run_dir / "raw" / "cvar.json", args)
                    run_dir.mkdir(parents=True, exist_ok=False)
                    elapsed = _run(_command(point, run_dir, deterministic, cvar, args), args.project_root, run_dir / "benchmark_driver.log", args.echo)
                rows.append(_row(point, repeat, run_dir, elapsed))
            except Exception as exc:
                rows.append({
                    "point": point.label, "axis": point.axis, "case": point.case,
                    "requested_value": point.requested_value, "actual_size": point.requested_value,
                    "repeat": repeat, "status": "failed", "run_directory": str(run_dir.resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                _write_outputs(rows, run_root, args.profile, plan)
                if not args.continue_on_error:
                    raise
    evidence = _write_outputs(rows, run_root, args.profile, plan)
    print(f"\nScalability evidence: {run_root}")
    print(evidence["claim_assessment"]["interpretation"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_scalability_updated(argv)


# Retain the old callable name for code that imports it directly.
main_scalability = main_scalability_updated


if __name__ == "__main__":
    raise SystemExit(main_scalability_updated())
