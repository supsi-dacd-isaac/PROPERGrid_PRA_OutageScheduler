"""Cost--risk frontier for deterministic and risk-informed outage schedules.

For every requested load multiplier, the script solves the deterministic
scheduler once and performs one broad, diverse CVaR search.  It then selects
the minimum in-sample CVaR candidate at each maintenance-utility tolerance and
validates every unique selected schedule on one common independent scenario
bank.  The validation also reconstructs preventive generation cost from the
normal-state dispatch returned by the SCOPF oracle.

Outputs comprise machine-readable JSON/CSV, Markdown and LaTeX tables, solver
logs, configuration snapshots, and cost--risk figures.  DNS exposure is not
called formal EENS because contingency occurrence probabilities are not part
of the current scheduler data model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--project-root", type=Path, default=PROJECT_ROOT)
    add("--case", required=True, type=Path, help="Network/scheduler JSON configuration.")
    add("--case-label", default="case")
    add("--deterministic-config", type=Path, default=PACKAGE_DIR / "config_clustered_example.json")
    add("--cvar-config", type=Path, default=PACKAGE_DIR / "config_clustered_cvar_example.json")
    add("--load-multipliers", nargs="+", type=float, default=(1.0,))
    add("--utility-tolerances", nargs="+", type=float, default=(0.0, 0.005, 0.01, 0.02, 0.05))
    add("--absolute-utility-tolerance", type=float, default=0.0)
    add("--scenario-count", type=int, default=48, help="CVaR optimisation scenarios.")
    add("--scenario-seed", type=int, default=42)
    add("--validation-samples", type=int, default=192)
    add("--validation-seed", type=int, default=20260727)
    add("--validation-workers", type=int, default=4)
    add("--critical-states", type=int, default=3)
    add("--cvar-alpha", type=float, default=0.95)
    add("--max-candidates", type=int, default=16)
    add("--patience", type=int, default=16)
    add("--min-candidate-changes", type=int, default=3)
    add("--oracle-workers", type=int, default=4)
    add("--master-threads", type=int, default=8)
    add("--contingency-top-k", type=int)
    add("--deterministic-max-candidates", type=int)
    add("--time-step-hours", type=float, default=1.0)
    add("--cost-unit", default="model cost units")
    add("--risk-equivalence-relative-tolerance", type=float, default=0.01)
    add("--risk-equivalence-absolute-tolerance", type=float, default=1e-6)
    add("--output-dir", type=Path)
    add("--resume", action="store_true")
    add("--dry-run", action="store_true")
    add("--echo", action="store_true")
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _write(path: Path, value: Any) -> None:
    from .serialization import json_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def _factor_label(value: float) -> str:
    return f"load_{value:.3f}".replace(".", "p")


def _variant(
    source: Path,
    section: str,
    destination: Path,
    result_path: Path,
    load_multiplier: float,
    args: argparse.Namespace,
) -> Path:
    document = _read(source)
    settings = dict(document.get(section, document))
    common = {
        "results_path": str(result_path),
        "iis_path": str(result_path.with_suffix(".ilp")),
        "oracle_workers": args.oracle_workers,
        "master_threads": args.master_threads,
        "critical_state_count": args.critical_states,
    }
    if args.contingency_top_k is not None:
        common["contingency_top_k"] = args.contingency_top_k
    settings.update(common)
    if section == "clustered_deterministic" and args.deterministic_max_candidates is not None:
        settings["max_candidates"] = args.deterministic_max_candidates
    if section == "clustered_cvar":
        settings.update({
            "scenario_count": args.scenario_count,
            "gaussian_samples_per_critical_state": None,
            "scenario_seed": args.scenario_seed,
            "cvar_alpha": args.cvar_alpha,
            "selection_rule": "utility_constrained_cvar",
            "relative_utility_tolerance": max(args.utility_tolerances),
            "absolute_utility_tolerance": args.absolute_utility_tolerance,
            "max_candidates": args.max_candidates,
            "patience": args.patience,
            "min_candidate_changes": args.min_candidate_changes,
        })
    document = {**document, section: settings} if section in document else {section: settings}
    document["stress_test"] = {"load_multiplier": load_multiplier}
    _write(destination, document)
    return destination


def _run_module(
    module: str,
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    echo: bool,
) -> float:
    merged = os.environ.copy()
    merged.update(environment)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=str(cwd),
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if echo:
                print(line, end="")
        code = process.wait()
    if code:
        raise RuntimeError(f"{module} failed with exit code {code}; see {log_path}.")
    return float(time.perf_counter() - started)


def _signature(candidate: Mapping[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return (
        tuple(sorted((str(key), str(value)) for key, value in (candidate.get("start_times") or {}).items())),
        tuple(sorted(str(value) for value in candidate.get("deferred_outages") or ())),
    )


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_map = {**{str(k): str(v) for k, v in (left.get("start_times") or {}).items()}, **{str(k): "<deferred>" for k in left.get("deferred_outages") or ()}}
    right_map = {**{str(k): str(v) for k, v in (right.get("start_times") or {}).items()}, **{str(k): "<deferred>" for k in right.get("deferred_outages") or ()}}
    return sum(left_map.get(key) != right_map.get(key) for key in set(left_map) | set(right_map))


def _frontier_selections(
    document: Mapping[str, Any],
    tolerances: Sequence[float],
    absolute_tolerance: float,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    candidates = list(document.get("candidates") or [])
    benchmark = next((item for item in candidates if str(item.get("source", "")).startswith("benchmark")), None)
    if benchmark is None:
        raise RuntimeError("The CVaR candidate pool does not contain the deterministic benchmark.")
    benchmark_utility = float(benchmark["maintenance_utility"])
    indexed = list(enumerate(candidates))
    selections: list[dict[str, Any]] = []
    for tolerance in sorted(set(float(value) for value in tolerances)):
        allowed_drop = max(absolute_tolerance, tolerance * max(abs(benchmark_utility), 1.0))
        floor = benchmark_utility - allowed_drop
        eligible = [
            (index, item) for index, item in indexed
            if bool(item.get("risk_feasible", True))
            and float(item.get("maintenance_utility", -math.inf)) + 1e-10 >= floor
        ]
        if not eligible:
            raise RuntimeError(f"No risk-feasible candidate satisfies utility tolerance {tolerance}.")
        index, selected = min(
            eligible,
            key=lambda pair: (
                float((pair[1].get("risk") or {}).get("cvar", math.inf)),
                float((pair[1].get("risk") or {}).get("expected_loss", math.inf)),
                -float(pair[1].get("maintenance_utility", -math.inf)),
                pair[0],
            ),
        )
        selections.append({
            "relative_utility_tolerance": tolerance,
            "utility_floor": floor,
            "pool_index": index,
            "candidate": selected,
        })
    return benchmark, selections


def _weighted_mean(values: Mapping[str, float], probabilities: Mapping[str, float]) -> float:
    total = float(sum(probabilities.values()))
    if total <= 0.0:
        raise ValueError("Scenario probabilities must have positive total mass.")
    return float(sum(float(probabilities[key]) * float(values[key]) for key in values) / total)


def _validate_schedules(
    data: Mapping[str, Any],
    schedules: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from .cluster_scopf_oracle import ClusterSCOPFOracle
    from .paired_risk_validation import PairedValidationConfig, active_outages_from_schedule, build_common_validation_samples
    from .risk_metrics import weighted_var_cvar
    from .scenario_sampling import demand_fingerprint

    config = PairedValidationConfig(
        scenario_count=args.validation_samples,
        scenario_seed=args.validation_seed,
        scenario_model="gaussian_critical_states",
        critical_state_count=args.critical_states,
        cvar_alpha=args.cvar_alpha,
        oracle_workers=args.validation_workers,
    )
    active = {
        label: active_outages_from_schedule(
            data,
            schedule.get("start_times") or {},
            schedule.get("deferred_outages") or (),
        )
        for label, schedule in schedules.items()
    }
    oracle = ClusterSCOPFOracle(data)
    samples, probabilities, representatives = build_common_validation_samples(data, config, oracle=oracle)
    all_contingencies = tuple(data["names"].get("contingencies", ()))
    generation_cost = {str(key): float(value) for key, value in (data.get("generation_cost") or {}).items()}
    generators = tuple(str(value) for value in data["names"].get("generators", ()))
    missing_costs = sorted(set(generators) - set(generation_cost))

    candidate_cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], Any] = {}
    baseline_cache: dict[tuple[str, tuple[str, ...]], Any] = {}
    cache_lock = threading.Lock()
    key_locks: dict[tuple[Any, ...], threading.Lock] = {}

    def cached_candidate(sample, outages, effective):
        demand_key = demand_fingerprint(sample.demand)
        key = (demand_key, tuple(sorted(outages)), tuple(effective))
        with cache_lock:
            cached = candidate_cache.get(key)
            lock = key_locks.setdefault(("candidate", *key), threading.Lock())
        if cached is None:
            with lock:
                with cache_lock:
                    cached = candidate_cache.get(key)
                if cached is None:
                    cached = oracle.solve(sample.source_time, outages, contingencies=effective, demand_override=sample.demand)
                    with cache_lock:
                        candidate_cache[key] = cached
        return cached

    def cached_baseline(sample, effective):
        demand_key = demand_fingerprint(sample.demand)
        key = (demand_key, tuple(effective))
        with cache_lock:
            cached = baseline_cache.get(key)
            lock = key_locks.setdefault(("baseline", *key), threading.Lock())
        if cached is None:
            with lock:
                with cache_lock:
                    cached = baseline_cache.get(key)
                if cached is None:
                    cached = oracle.solve(sample.source_time, (), contingencies=effective, demand_override=sample.demand)
                    with cache_lock:
                        baseline_cache[key] = cached
        return cached

    def solve_one(label: str, sample) -> dict[str, Any]:
        outages = active[label][sample.source_time]
        effective = oracle.effective_contingencies(outages, all_contingencies)
        candidate = cached_candidate(sample, outages, effective)
        baseline = cached_baseline(sample, effective)
        if not math.isfinite(candidate.maximum_dns) or not math.isfinite(baseline.maximum_dns):
            raise RuntimeError(f"Validation SCOPF failed for {label}/{sample.scenario_id}.")
        incremental_max = max(0.0, float(candidate.maximum_dns - baseline.maximum_dns))
        incremental_total = max(0.0, float(candidate.total_dns - baseline.total_dns))
        mean_incremental = incremental_total / max(1, len(candidate.contingencies) + 1)
        loss = incremental_max + config.total_dns_component_weight * mean_incremental
        normal = candidate.normal_state
        operating_cost = args.time_step_hours * sum(
            generation_cost.get(generator, 1.0) * float(output)
            for generator, output in normal.generation.items()
        )
        return {
            "schedule": label,
            "scenario_id": sample.scenario_id,
            "probability": float(sample.probability),
            "source_time": sample.source_time,
            "load_multiplier": float(data.get("load_multiplier", 1.0)),
            "total_demand_mw": float(sample.total_demand),
            "normal_state_operating_cost": float(operating_cost),
            "normal_state_dns_mw": float(normal.load_shedding),
            "candidate_maximum_dns_mw": float(candidate.maximum_dns),
            "candidate_total_dns_mw": float(candidate.total_dns),
            "incremental_maximum_dns_mw": incremental_max,
            "incremental_total_dns_mw": incremental_total,
            "composite_incremental_loss_mw": float(loss),
            "worst_contingency": candidate.worst_contingency,
        }

    tasks = [(label, sample) for sample in samples for label in schedules]
    records: list[dict[str, Any]] = []
    workers = max(1, int(args.validation_workers))
    if workers == 1:
        records = [solve_one(label, sample) for label, sample in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(solve_one, label, sample) for label, sample in tasks]
            records = [future.result() for future in as_completed(futures)]
    records.sort(key=lambda item: (item["scenario_id"], item["schedule"]))

    metrics: dict[str, dict[str, Any]] = {}
    for label in schedules:
        selected = [record for record in records if record["schedule"] == label]
        losses = {record["scenario_id"]: record["composite_incremental_loss_mw"] for record in selected}
        risk = weighted_var_cvar(losses, probabilities, alpha=args.cvar_alpha)
        costs = {record["scenario_id"]: record["normal_state_operating_cost"] for record in selected}
        normal_dns = {record["scenario_id"]: record["normal_state_dns_mw"] for record in selected}
        worst_dns = {record["scenario_id"]: record["candidate_maximum_dns_mw"] for record in selected}
        metrics[label] = {
            "risk": asdict(risk),
            "maximum_composite_loss_mw": max(losses.values(), default=0.0),
            "probability_positive_incremental_loss": sum(probabilities[key] for key, value in losses.items() if value > 1e-9) / sum(probabilities.values()),
            "expected_operating_cost_per_timestep": _weighted_mean(costs, probabilities),
            "expected_normal_state_dns_mw": _weighted_mean(normal_dns, probabilities),
            "expected_normal_state_unserved_energy_mwh_per_timestep": _weighted_mean(normal_dns, probabilities) * args.time_step_hours,
            "expected_worst_contingency_dns_mw": _weighted_mean(worst_dns, probabilities),
        }
    return {
        "design": {
            "config": asdict(config),
            "load_multiplier": float(data.get("load_multiplier", 1.0)),
            "schedule_count": len(schedules),
            "scenario_probabilities": probabilities,
            "critical_validation_states": representatives,
            "cost_definition": "Sample-weighted preventive normal-state generation dispatch multiplied by configured generator marginal costs and time-step duration.",
            "dns_note": "DNS exposure is not formal EENS because contingency occurrence probabilities are unavailable.",
            "generation_cost_complete": not missing_costs,
            "generators_with_default_unit_cost": missing_costs,
        },
        "metrics": metrics,
        "records": records,
        "performance": {
            "oracle_statistics": oracle.statistics(),
            "candidate_cache_entries": len(candidate_cache),
            "baseline_cache_entries": len(baseline_cache),
        },
    }


def _percent_change(value: float, baseline: float) -> float | None:
    return 100.0 * (value - baseline) / abs(baseline) if abs(baseline) > 1e-12 else None


def _is_pareto(rows: Sequence[Mapping[str, Any]], index: int) -> bool:
    target = rows[index]
    cost, risk = float(target["validation_expected_operating_cost"]), float(target["validation_cvar_mw"])
    for other_index, other in enumerate(rows):
        if other_index == index:
            continue
        other_cost, other_risk = float(other["validation_expected_operating_cost"]), float(other["validation_cvar_mw"])
        if other_cost <= cost + 1e-10 and other_risk <= risk + 1e-10 and (other_cost < cost - 1e-10 or other_risk < risk - 1e-10):
            return False
    return True


def _rows_for_load(
    load_multiplier: float,
    benchmark: Mapping[str, Any],
    selections: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    det_metrics = validation["metrics"]["deterministic"]
    det_cost = float(det_metrics["expected_operating_cost_per_timestep"])
    det_cvar = float(det_metrics["risk"]["cvar"])
    benchmark_utility = float(benchmark["maintenance_utility"])
    rows: list[dict[str, Any]] = []

    def build(label: str, formulation: str, tolerance: float | None, candidate: Mapping[str, Any], pool_index: int | None) -> dict[str, Any]:
        metrics = validation["metrics"][label]
        cost = float(metrics["expected_operating_cost_per_timestep"])
        cvar = float(metrics["risk"]["cvar"])
        utility = float(candidate["maintenance_utility"])
        equivalence_band = max(args.risk_equivalence_absolute_tolerance, args.risk_equivalence_relative_tolerance * max(abs(det_cvar), 1e-12))
        return {
            "case": args.case_label,
            "load_multiplier": load_multiplier,
            "formulation": formulation,
            "relative_utility_tolerance": tolerance,
            "pool_index": pool_index,
            "candidate_source": candidate.get("source"),
            "maintenance_utility": utility,
            "utility_loss": benchmark_utility - utility,
            "utility_loss_percent": 100.0 * (benchmark_utility - utility) / max(abs(benchmark_utility), 1.0),
            "schedule_distance_from_deterministic": _distance(benchmark, candidate),
            "training_expected_loss_mw": float((candidate.get("risk") or {}).get("expected_loss", math.nan)),
            "training_var_mw": float((candidate.get("risk") or {}).get("var", math.nan)),
            "training_cvar_mw": float((candidate.get("risk") or {}).get("cvar", math.nan)),
            "validation_expected_loss_mw": float(metrics["risk"]["expected_loss"]),
            "validation_var_mw": float(metrics["risk"]["var"]),
            "validation_cvar_mw": cvar,
            "validation_maximum_loss_mw": float(metrics["maximum_composite_loss_mw"]),
            "probability_positive_incremental_loss": float(metrics["probability_positive_incremental_loss"]),
            "validation_expected_operating_cost": cost,
            "operating_cost_change": cost - det_cost,
            "operating_cost_change_percent": _percent_change(cost, det_cost),
            "expected_normal_state_unserved_energy_mwh_per_timestep": float(metrics["expected_normal_state_unserved_energy_mwh_per_timestep"]),
            "expected_worst_contingency_dns_mw": float(metrics["expected_worst_contingency_dns_mw"]),
            "cvar_change_mw": cvar - det_cvar,
            "cvar_reduction_percent": -_percent_change(cvar, det_cvar) if _percent_change(cvar, det_cvar) is not None else None,
            "risk_equivalent_to_deterministic": abs(cvar - det_cvar) <= equivalence_band,
            "risk_not_higher_than_deterministic": cvar <= det_cvar + equivalence_band,
            "dominates_deterministic": cost <= det_cost + 1e-10 and cvar <= det_cvar + 1e-10 and (cost < det_cost - 1e-10 or cvar < det_cvar - 1e-10),
        }

    rows.append(build("deterministic", "deterministic", None, benchmark, None))
    for selection in selections:
        candidate = selection["candidate"]
        rows.append(build(f"candidate_{selection['pool_index']}", "risk_informed", float(selection["relative_utility_tolerance"]), candidate, int(selection["pool_index"])))
    for index, row in enumerate(rows):
        row["pareto_efficient"] = _is_pareto(rows, index)
    return rows


def _display(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "--" if not math.isfinite(value) else f"{value:.5g}"
    return str(value)


TABLE = (
    ("load_multiplier", "Load"),
    ("formulation", "Method"),
    ("relative_utility_tolerance", "Utility tol."),
    ("schedule_distance_from_deterministic", "Distance"),
    ("utility_loss_percent", "Utility loss [%]"),
    ("validation_expected_operating_cost", "Expected cost"),
    ("operating_cost_change_percent", "Cost change [%]"),
    ("validation_expected_loss_mw", "Expected loss [MW]"),
    ("validation_var_mw", "VaR [MW]"),
    ("validation_cvar_mw", "CVaR [MW]"),
    ("cvar_reduction_percent", "CVaR reduction [%]"),
    ("risk_not_higher_than_deterministic", "Risk no higher"),
    ("pareto_efficient", "Pareto"),
)


def _latex(value: Any) -> str:
    text = _display(value)
    for source, target in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(source, target)
    return text


def _plot(rows: Sequence[Mapping[str, Any]], output: Path, cost_unit: str, alpha: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    loads = sorted({float(row["load_multiplier"]) for row in rows})
    colors = plt.cm.viridis([index / max(1, len(loads) - 1) for index in range(len(loads))])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for load, color in zip(loads, colors):
        selected = [row for row in rows if float(row["load_multiplier"]) == load]
        det = next(row for row in selected if row["formulation"] == "deterministic")
        risk_rows = [row for row in selected if row["formulation"] == "risk_informed"]
        axes[0].scatter(det["validation_expected_operating_cost"], det["validation_cvar_mw"], marker="s", s=85, color=color, edgecolor="black", label=f"load {load:.2f}")
        axes[0].plot([row["validation_expected_operating_cost"] for row in risk_rows], [row["validation_cvar_mw"] for row in risk_rows], "o-", color=color, alpha=.85)
        axes[1].scatter(0.0, 0.0, marker="s", s=85, color=color, edgecolor="black")
        axes[1].plot([row["utility_loss_percent"] for row in risk_rows], [row["cvar_reduction_percent"] or 0.0 for row in risk_rows], "o-", color=color, alpha=.85, label=f"load {load:.2f}")
    axes[0].set(xlabel=f"Expected preventive operating cost [{cost_unit}/time step]", ylabel=f"Validation CVaR$_{{{alpha:.2f}}}$ [MW]", title="Operating cost--risk frontier")
    axes[1].axhline(0.0, color="black", linewidth=.8)
    axes[1].set(xlabel="Maintenance-utility loss [%]", ylabel="CVaR reduction relative to deterministic [%]", title="Utility--risk trade-off")
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(root: Path, rows: list[dict[str, Any]], evidence: Mapping[str, Any], args: argparse.Namespace) -> None:
    _write(root / "cost_risk_results.json", evidence)
    fields = sorted({key for row in rows for key in row})
    with (root / "cost_risk_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    headers = [label for _, label in TABLE]
    lines = [
        "# WP3 cost--risk trade-off", "",
        f"Case: **{args.case_label}**; validation scenarios: **{args.validation_samples}**; CVaR level: **{args.cvar_alpha:.3g}**.", "",
        "The expected operating-cost KPI is the sample-weighted preventive normal-state dispatch cost. DNS is reported as exposure rather than formal EENS because contingency occurrence probabilities are not available.", "",
        "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(_display(row.get(key)) for key, _ in TABLE) + " |" for row in rows],
        "", "## Interpretation", "",
    ]
    for load in sorted({float(row["load_multiplier"]) for row in rows}):
        group = [row for row in rows if float(row["load_multiplier"]) == load]
        dominating = [row for row in group if row["formulation"] == "risk_informed" and row["dominates_deterministic"]]
        if dominating:
            best = min(dominating, key=lambda row: (row["validation_expected_operating_cost"], row["validation_cvar_mw"]))
            lines.append(f"- Load multiplier {load:.3g}: a risk-informed point dominates the deterministic schedule (utility tolerance {_display(best['relative_utility_tolerance'])}, cost change {_display(best['operating_cost_change_percent'])}%, CVaR reduction {_display(best['cvar_reduction_percent'])}%).")
        else:
            lines.append(f"- Load multiplier {load:.3g}: no evaluated risk-informed point simultaneously reduced expected operating cost and validation CVaR.")
    lines += ["", "Only independent common-sample validation metrics should support superiority claims; training CVaR is diagnostic.", ""]
    (root / "cost_risk_report.md").write_text("\n".join(lines), encoding="utf-8")
    latex = [
        r"\begin{table}[t]", r"\centering\scriptsize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * len(TABLE) + "}", r"\toprule",
        " & ".join(_latex(label) for label in headers) + r" \\", r"\midrule",
        *[" & ".join(_latex(row.get(key)) for key, _ in TABLE) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\caption{Independent common-sample cost--risk comparison. Expected cost denotes preventive normal-state dispatch cost; DNS exposure is not formal EENS without contingency occurrence probabilities.}",
        r"\label{tab:wp3_cost_risk_tradeoff}", r"\end{table}",
    ]
    (root / "cost_risk_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    _plot(rows, root / "cost_risk_frontier.png", args.cost_unit, args.cvar_alpha)


def _validate_args(args: argparse.Namespace) -> None:
    if any(not math.isfinite(value) or value <= 0.0 for value in args.load_multipliers):
        raise ValueError("All load multipliers must be finite and positive.")
    if any(not 0.0 <= value < 1.0 for value in args.utility_tolerances):
        raise ValueError("Utility tolerances must lie in [0, 1).")
    if args.scenario_count < 2 or args.validation_samples < 2:
        raise ValueError("Scenario counts must be at least two.")
    if not 0.0 < args.cvar_alpha < 1.0:
        raise ValueError("--cvar-alpha must lie in (0, 1).")
    if min(args.max_candidates, args.patience, args.min_candidate_changes, args.validation_workers) < 1:
        raise ValueError("Candidate, patience, distance and worker settings must be positive.")
    if args.time_step_hours <= 0.0:
        raise ValueError("--time-step-hours must be positive.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    args.project_root = args.project_root.expanduser().resolve()
    network = _resolve(args.project_root, args.case)
    deterministic_base = _resolve(args.project_root, args.deterministic_config)
    cvar_base = _resolve(args.project_root, args.cvar_config)
    for path in (network, deterministic_base, cvar_base):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir is None:
        root = args.project_root / "outputs" / "wp3_cost_risk" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        root = _resolve(args.project_root, args.output_dir)
    if root.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists; use --resume to reuse it: {root}")
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "case": str(network),
        "deterministic_config": str(deterministic_base),
        "cvar_config": str(cvar_base),
        "load_multipliers": sorted(set(args.load_multipliers)),
        "utility_tolerances": sorted(set(args.utility_tolerances)),
        "absolute_utility_tolerance": args.absolute_utility_tolerance,
        "optimisation_scenarios": args.scenario_count,
        "optimisation_seed": args.scenario_seed,
        "validation_scenarios": args.validation_samples,
        "validation_seed": args.validation_seed,
        "cvar_alpha": args.cvar_alpha,
        "candidate_budget": args.max_candidates,
        "deterministic_candidate_budget_override": args.deterministic_max_candidates,
        "patience": args.patience,
        "minimum_candidate_changes": args.min_candidate_changes,
        "critical_states": args.critical_states,
        "contingency_top_k": args.contingency_top_k,
        "time_step_hours": args.time_step_hours,
        "cost_unit": args.cost_unit,
    }
    plan_path = root / "study_plan.json"
    if args.resume and plan_path.is_file() and _read(plan_path) != plan:
        raise ValueError("The existing study plan differs from the requested --resume settings.")
    _write(plan_path, plan)
    if args.dry_run:
        print(f"Dry-run plan written to {root}")
        return 0

    from .paired_risk_validation import load_validation_data

    all_rows: list[dict[str, Any]] = []
    load_results: list[dict[str, Any]] = []
    for load_multiplier in sorted(set(args.load_multipliers)):
        label = _factor_label(load_multiplier)
        point = root / label
        inputs, results, logs = point / "inputs", point / "results", point / "logs"
        results.mkdir(parents=True, exist_ok=True)
        det_result, cvar_result = results / "clustered_deterministic_results.json", results / "clustered_cvar_results.json"
        det_config = _variant(deterministic_base, "clustered_deterministic", inputs / "deterministic.json", det_result, load_multiplier, args)
        cvar_config = _variant(cvar_base, "clustered_cvar", inputs / "cvar.json", cvar_result, load_multiplier, args)
        environment = {
            "PROPER_SCHEDULER_CONFIG": str(network),
            "PROPER_CLUSTERED_CONFIG": str(det_config),
            "PROPER_CLUSTERED_CVAR_CONFIG": str(cvar_config),
        }
        timings: dict[str, float] = {}
        if not (args.resume and det_result.is_file()):
            print(f"\n=== {label}: deterministic optimisation ===")
            timings["deterministic_seconds"] = _run_module("scheduler_clustered.main_clustered", cwd=args.project_root, environment=environment, log_path=logs / "deterministic.log", echo=args.echo)
        if not (args.resume and cvar_result.is_file()):
            print(f"\n=== {label}: diverse CVaR candidate pool ===")
            timings["cvar_seconds"] = _run_module("scheduler_clustered.main_clustered_cvar", cwd=args.project_root, environment={**environment, "PROPER_DETERMINISTIC_RESULTS": str(det_result)}, log_path=logs / "cvar.log", echo=args.echo)

        deterministic, cvar = _read(det_result), _read(cvar_result)
        benchmark, selections = _frontier_selections(cvar, args.utility_tolerances, args.absolute_utility_tolerance)
        if dict(benchmark.get("start_times") or {}) != dict(deterministic.get("best_schedule") or {}):
            raise RuntimeError("The CVaR benchmark does not match the deterministic result.")
        schedules: dict[str, Mapping[str, Any]] = {"deterministic": benchmark}
        for selection in selections:
            schedules.setdefault(f"candidate_{selection['pool_index']}", selection["candidate"])
        print(f"\n=== {label}: common validation of {len(schedules)} unique schedules ===")
        started = time.perf_counter()
        data = load_validation_data(network, cvar_config)
        validation = _validate_schedules(data, schedules, args)
        timings["validation_seconds"] = float(time.perf_counter() - started)
        _write(results / "frontier_validation.json", validation)
        rows = _rows_for_load(load_multiplier, benchmark, selections, validation, args)
        all_rows.extend(rows)
        load_results.append({
            "load_multiplier": load_multiplier,
            "deterministic_results": str(det_result),
            "cvar_results": str(cvar_result),
            "selected_pool_indices": [int(item["pool_index"]) for item in selections],
            "unique_validated_schedules": len(schedules),
            "timings": timings,
            "validation": validation,
        })
        _write_outputs(root, all_rows, {"plan": plan, "rows": all_rows, "load_results": load_results}, args)

    print(f"\nCost--risk study written to: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
