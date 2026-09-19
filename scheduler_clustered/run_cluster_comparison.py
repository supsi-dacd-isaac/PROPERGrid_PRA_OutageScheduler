"""Run deterministic and CVaR schedulers and perform paired risk validation.

The primary deterministic-versus-CVaR risk conclusion is produced only after
both final schedules have been evaluated on one common operating-state sample
bank. Formulation-specific objective scores and unpaired cluster-severity
plots are not used to decide which schedule is less risky.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

# This file lives in PROPER/scheduler_clustered/. The project root is its parent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MPLBACKEND", "Agg")

from .comparison_visualization import generate_unambiguous_comparison
from .results_visualization import generate_all_plots


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    print(f"\n=== Running {' '.join(command)} ===")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_environment = os.environ.copy()
    if environment:
        merged_environment.update(environment)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. See {log_path}."
        )
    return {
        "command": list(command),
        "wall_time_seconds": float(elapsed),
        "return_code": int(return_code),
        "log": str(log_path.resolve()),
    }


def _child_peak_rss_mb() -> float | None:
    """Best-effort maximum resident set size of completed child processes."""
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if value <= 0.0:
            return None
        # Linux reports KiB; macOS reports bytes.
        divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
        return value / divisor
    except (ImportError, AttributeError, OSError):
        return None


def _software_environment() -> dict[str, object]:
    gurobi_version = None
    try:
        import gurobipy

        gurobi_version = ".".join(str(value) for value in gurobipy.gurobi.version())
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "gurobi": gurobi_version,
    }


def _snapshot(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Expected result file not found: {source}")
    if source.resolve() == destination.resolve():
        return source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _result_path(project_root: Path, config_path: Path, section: str, default: str) -> Path:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    configured = (document.get(section, document) or {}).get("results_path", default)
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _option(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--project-root", type=Path, default=PROJECT_ROOT, help="PROPER project root.")
    add("--config", type=Path, default=Path(os.getenv("PROPER_SCHEDULER_CONFIG", "config/conf_IEEE24_scheduler_v2.json")), help="Network/scheduler configuration.")
    add("--deterministic-config", type=Path, help="Clustered deterministic JSON configuration.")
    add("--cvar-config", type=Path, help="Clustered CVaR JSON configuration.")
    add("--output-dir", type=Path, help="Output directory; default: timestamped run under outputs/.")
    add("--reuse-results", action="store_true", help="Reuse existing deterministic/CVaR JSON files.")
    add("--reuse-paired-validation", type=Path, help="Reuse an existing paired_risk_validation.json.")
    add("--allow-missing-benchmark", action="store_true", help="Permit non-evidentiary diagnostics without the deterministic CVaR benchmark.")
    add("--show", action="store_true")
    add("--skip-plots", action="store_true", help="Skip plot generation; useful for timings.")
    add("--skip-validation", action="store_true", help="Skip paired validation; timing evidence only.")
    add("--deterministic-module", default="scheduler_clustered.main_clustered")
    add("--cvar-module", default="scheduler_clustered.main_clustered_cvar")
    add("--validation-samples", type=int, default=96)
    add("--validation-scenario-model", choices=("gaussian_critical_states", "empirical_cluster"), default="gaussian_critical_states")
    add("--validation-critical-states", type=int, default=3)
    add("--validation-seed", type=int, default=20260727)
    add("--validation-sampling-scheme", choices=("random", "stratified_tail"), default="random")
    add("--validation-sigma", type=float, help="Legacy empirical global lognormal sigma.")
    add("--validation-relative-sigma", type=float)
    add("--validation-global-sigma", type=float)
    add("--validation-alpha", type=float)
    add("--validation-workers", type=int, default=1)
    add("--winner-tolerance-mw", type=float, default=1e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    workflow_started = time.perf_counter()
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    if not (project_root / "scheduler_clustered").is_dir():
        raise FileNotFoundError(
            f"scheduler_clustered package not found under {project_root}."
        )

    config_path = _resolve(project_root, args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Scheduler configuration not found: {config_path}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            project_root
            / "outputs"
            / "clustered_comparison_runs"
            / f"run_{stamp}"
        )
    else:
        run_dir = _resolve(project_root, args.output_dir)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    scheduler_environment = {
        "PROPER_SCHEDULER_CONFIG": str(config_path),
    }
    deterministic_config_path = _resolve(project_root, args.deterministic_config or Path("scheduler_clustered/config_clustered_example.json"))
    cvar_config_path = _resolve(project_root, args.cvar_config or Path("scheduler_clustered/config_clustered_cvar_example.json"))
    scheduler_environment.update({"PROPER_CLUSTERED_CONFIG": str(deterministic_config_path), "PROPER_CLUSTERED_CVAR_CONFIG": str(cvar_config_path)})
    for required_config in (config_path, deterministic_config_path, cvar_config_path):
        if not required_config.is_file():
            raise FileNotFoundError(required_config)
    deterministic_source = _result_path(
        project_root,
        deterministic_config_path,
        "clustered_deterministic",
        "clustered_deterministic_results.json",
    )
    cvar_source = _result_path(
        project_root,
        cvar_config_path,
        "clustered_cvar",
        "clustered_cvar_results.json",
    )

    input_snapshots = {
        "network_config": str(
            _snapshot(config_path, run_dir / "inputs" / config_path.name)
        ),
        "deterministic_config": str(
            _snapshot(
                deterministic_config_path,
                run_dir / "inputs" / deterministic_config_path.name,
            )
        ),
        "cvar_config": str(
            _snapshot(cvar_config_path, run_dir / "inputs" / cvar_config_path.name)
        ),
    }

    stage_performance: dict[str, object] = {}
    if not args.reuse_results:
        stage_performance["deterministic_optimization"] = _run_command(
            [sys.executable, "-m", args.deterministic_module],
            cwd=project_root,
            log_path=run_dir / "logs" / "deterministic.log",
            environment=scheduler_environment,
        )
        cvar_environment = dict(scheduler_environment)
        cvar_environment["PROPER_DETERMINISTIC_RESULTS"] = str(
            deterministic_source
        )
        stage_performance["cvar_optimization"] = _run_command(
            [sys.executable, "-m", args.cvar_module],
            cwd=project_root,
            log_path=run_dir / "logs" / "cvar.log",
            environment=cvar_environment,
        )

    deterministic_snapshot = _snapshot(
        deterministic_source,
        run_dir / "results" / "clustered_deterministic_results.json",
    )
    cvar_snapshot = _snapshot(
        cvar_source,
        run_dir / "results" / "clustered_cvar_results.json",
    )
    deterministic_document = json.loads(
        deterministic_snapshot.read_text(encoding="utf-8")
    )
    cvar_document = json.loads(cvar_snapshot.read_text(encoding="utf-8"))
    if (
        not cvar_document.get("benchmark_included", False)
        and not args.allow_missing_benchmark
    ):
        raise RuntimeError(
            "The CVaR result does not include the deterministic benchmark "
            "schedule. Rerun both optimisers through this driver, or use "
            "--allow-missing-benchmark only for non-evidentiary diagnostics."
        )

    if not args.skip_plots:
        generate_all_plots(deterministic_snapshot, output_dir=run_dir / "visuals" / "deterministic", show=args.show)
        generate_all_plots(cvar_snapshot, output_dir=run_dir / "visuals" / "cvar", show=args.show)

    paired_path = run_dir / "results" / "paired_risk_validation.json"
    if args.skip_validation:
        paired_path = None
        stage_performance["paired_validation"] = {"mode": "skipped"}
    elif args.reuse_paired_validation is not None:
        validation_started = time.perf_counter()
        paired_path = _snapshot(_resolve(project_root, args.reuse_paired_validation), paired_path)
        stage_performance["paired_validation"] = {
            "wall_time_seconds": float(time.perf_counter() - validation_started),
            "mode": "reused",
        }
    else:
        validation_command = [
            sys.executable,
            "-m",
            "scheduler_clustered.paired_risk_validation",
            "--deterministic-results",
            str(deterministic_snapshot),
            "--cvar-results",
            str(cvar_snapshot),
            "--config",
            str(config_path),
            "--output",
            str(paired_path),
            "--data-overrides-config",
            str(cvar_config_path),
            "--scenario-count",
            str(args.validation_samples),
            "--scenario-seed",
            str(args.validation_seed),
            "--scenario-model",
            str(args.validation_scenario_model),
            "--critical-state-count",
            str(args.validation_critical_states),
            "--sampling-scheme",
            str(args.validation_sampling_scheme),
            "--oracle-workers",
            str(args.validation_workers),
            "--winner-tolerance-mw",
            str(args.winner_tolerance_mw),
        ]
        _option(validation_command, "--global-lognormal-sigma", args.validation_sigma)
        _option(validation_command, "--gaussian-relative-sigma", args.validation_relative_sigma)
        _option(validation_command, "--gaussian-global-sigma", args.validation_global_sigma)
        _option(validation_command, "--cvar-alpha", args.validation_alpha)
        stage_performance["paired_validation"] = _run_command(
            validation_command,
            cwd=project_root,
            log_path=run_dir / "logs" / "paired_validation.log",
            environment=scheduler_environment,
        )

    comparison_dir = None
    paired_performance = None
    verdict = {
        "tail_risk_winner": None,
        "conclusion": "Paired validation was skipped; this run provides computational timing only.",
    }
    if paired_path is not None:
        if not args.skip_plots:
            comparison_dir = generate_unambiguous_comparison(
                deterministic_snapshot,
                cvar_snapshot,
                paired_path,
                output_dir=run_dir / "visuals" / "comparison",
                show=args.show,
            )
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        paired_performance = paired.get("performance")
        verdict = paired["comparison"]
    dimensions: dict[str, object]
    try:
        from .data_adapter import power_data_diagnostics
        from .runtime import (
            apply_data_overrides_from_json,
            configured_decomposition_data,
        )

        dimension_data = configured_decomposition_data(
            config_path, allow_outage_deferral=True
        )
        dimension_data = apply_data_overrides_from_json(
            dimension_data,
            str(cvar_config_path) if cvar_config_path is not None else None,
        )
        diagnostics = power_data_diagnostics(dimension_data)
        dimensions = {
            "buses": int(diagnostics["buses"]),
            "branches": int(diagnostics["branches"]),
            "generators": int(diagnostics["generators_represented"]),
            "planned_outages": len(dimension_data["names"]["outages"]),
            "n_minus_1_contingencies": len(
                dimension_data["names"]["contingencies"]
            ),
            "horizon_steps": len(dimension_data["T"]),
            "benchmark_slice": dimension_data.get("benchmark_slice"),
        }
    except Exception as exc:
        dimensions = {"error": f"{type(exc).__name__}: {exc}"}

    workflow_wall_time = float(time.perf_counter() - workflow_started)
    performance = {
        "stages": stage_performance,
        "workflow_wall_time_seconds": workflow_wall_time,
        "child_peak_rss_mb": _child_peak_rss_mb(),
        "software_environment": _software_environment(),
        "problem_dimensions": dimensions,
        "deterministic": {
            "oracle_statistics": deterministic_document.get("oracle_statistics"),
            "iterations": deterministic_document.get("iterations"),
            "candidate_count": len(deterministic_document.get("candidates") or []),
            "termination_reason": deterministic_document.get("termination_reason"),
            "max_candidates": (deterministic_document.get("config") or {}).get(
                "max_candidates"
            ),
            "critical_state_count": (deterministic_document.get("config") or {}).get(
                "critical_state_count"
            ),
            "master_threads": (deterministic_document.get("config") or {}).get(
                "master_threads"
            ),
            "oracle_workers": (deterministic_document.get("config") or {}).get(
                "oracle_workers"
            ),
            "master_time_limit_seconds": (
                deterministic_document.get("config") or {}
            ).get("master_time_limit"),
            "master_mip_gap": (deterministic_document.get("config") or {}).get(
                "master_mip_gap"
            ),
            "contingency_top_k": (deterministic_document.get("config") or {}).get(
                "contingency_top_k"
            ),
            "full_contingency_validation": (
                deterministic_document.get("config") or {}
            ).get("full_contingency_validation"),
        },
        "cvar": {
            "oracle_statistics": cvar_document.get("oracle_statistics"),
            "iterations": cvar_document.get("iterations"),
            "candidate_count": len(cvar_document.get("candidates") or []),
            "termination_reason": cvar_document.get("termination_reason"),
            "scenario_count": (cvar_document.get("config") or {}).get(
                "scenario_count"
            ),
            "max_candidates": (cvar_document.get("config") or {}).get(
                "max_candidates"
            ),
            "critical_state_count": (cvar_document.get("config") or {}).get(
                "critical_state_count"
            ),
            "master_threads": (cvar_document.get("config") or {}).get(
                "master_threads"
            ),
            "oracle_workers": (cvar_document.get("config") or {}).get(
                "oracle_workers"
            ),
            "master_time_limit_seconds": (cvar_document.get("config") or {}).get(
                "master_time_limit"
            ),
            "master_mip_gap": (cvar_document.get("config") or {}).get(
                "master_mip_gap"
            ),
            "contingency_top_k": (cvar_document.get("config") or {}).get(
                "contingency_top_k"
            ),
            "full_contingency_validation": (cvar_document.get("config") or {}).get(
                "full_contingency_validation"
            ),
            "benchmark_included": cvar_document.get("benchmark_included"),
        },
        "validation": {
            "enabled": not args.skip_validation,
            "scenario_count": int(args.validation_samples),
            "critical_state_count": int(args.validation_critical_states),
            "scenario_seed": int(args.validation_seed),
            "scenario_model": args.validation_scenario_model,
            "sampling_scheme": args.validation_sampling_scheme,
            "cvar_alpha": args.validation_alpha,
            "oracle_workers": int(args.validation_workers),
            "performance": paired_performance,
        },
    }
    (run_dir / "computational_performance.json").write_text(
        json.dumps(performance, indent=2, allow_nan=False), encoding="utf-8"
    )

    manifest = {
        "project_root": str(project_root),
        "config": str(config_path),
        "deterministic_results": str(deterministic_snapshot),
        "cvar_results": str(cvar_snapshot),
        "paired_validation": str(paired_path) if paired_path is not None else None,
        "comparison_visuals": str(comparison_dir) if comparison_dir is not None else None,
        "tail_risk_winner": verdict["tail_risk_winner"],
        "conclusion": verdict["conclusion"],
        "input_snapshots": input_snapshots,
        "deterministic_config": str(deterministic_config_path),
        "cvar_config": str(cvar_config_path),
        "performance": performance,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("\n=== Completed deterministic + CVaR paired comparison ===")
    print(f"Run directory: {run_dir}")
    print(f"Tail-risk winner: {verdict['tail_risk_winner'] or 'not evaluated'}")
    print(verdict["conclusion"])


def main_scalability(argv: Sequence[str] | None = None) -> int:
    """Run matched grid/outage/horizon experiments and write reviewer evidence."""
    from .scalability_study import main_scalability as run
    return run(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scalability":
        raise SystemExit(main_scalability(sys.argv[2:]))
    main()
