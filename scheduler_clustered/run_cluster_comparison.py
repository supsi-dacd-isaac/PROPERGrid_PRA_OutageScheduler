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
import shutil
import subprocess
import sys
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
) -> None:
    print(f"\n=== Running {' '.join(command)} ===")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_environment = os.environ.copy()
    if environment:
        merged_environment.update(environment)
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
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. See {log_path}."
        )


def _snapshot(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Expected result file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="PROPER project root.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.getenv(
                "PROPER_SCHEDULER_CONFIG",
                "config/conf_IEEE24_scheduler_v2.json",
            )
        ),
        help="Network/scheduler input configuration used by both formulations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Run output directory. By default, a timestamped directory is "
            "created under outputs/clustered_comparison_runs/."
        ),
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Skip deterministic and CVaR optimisation and use existing root JSON files.",
    )
    parser.add_argument(
        "--reuse-paired-validation",
        type=Path,
        default=None,
        help="Use an existing paired_risk_validation.json instead of rerunning SCOPF validation.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--deterministic-module",
        default="scheduler_clustered.main_clustered",
    )
    parser.add_argument(
        "--cvar-module",
        default="scheduler_clustered.main_clustered_cvar",
    )
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=20260727)
    parser.add_argument(
        "--validation-sampling-scheme",
        choices=("random", "stratified_tail"),
        default="random",
    )
    parser.add_argument("--validation-sigma", type=float, default=None)
    parser.add_argument("--validation-alpha", type=float, default=None)
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--winner-tolerance-mw", type=float, default=1e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    if not (project_root / "scheduler_clustered").is_dir():
        raise FileNotFoundError(
            f"scheduler_clustered package not found under {project_root}."
        )

    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
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
        run_dir = args.output_dir
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    deterministic_source = project_root / "clustered_deterministic_results.json"
    cvar_source = project_root / "clustered_cvar_results.json"
    scheduler_environment = {
        "PROPER_SCHEDULER_CONFIG": str(config_path),
    }

    if not args.reuse_results:
        _run_command(
            [sys.executable, "-m", args.deterministic_module],
            cwd=project_root,
            log_path=run_dir / "logs" / "deterministic.log",
            environment=scheduler_environment,
        )
        _run_command(
            [sys.executable, "-m", args.cvar_module],
            cwd=project_root,
            log_path=run_dir / "logs" / "cvar.log",
            environment=scheduler_environment,
        )

    deterministic_snapshot = _snapshot(
        deterministic_source,
        run_dir / "results" / "clustered_deterministic_results.json",
    )
    cvar_snapshot = _snapshot(
        cvar_source,
        run_dir / "results" / "clustered_cvar_results.json",
    )

    generate_all_plots(
        deterministic_snapshot,
        output_dir=run_dir / "visuals" / "deterministic",
        show=args.show,
    )
    generate_all_plots(
        cvar_snapshot,
        output_dir=run_dir / "visuals" / "cvar",
        show=args.show,
    )

    paired_path = run_dir / "results" / "paired_risk_validation.json"
    if args.reuse_paired_validation is not None:
        paired_path = _snapshot(args.reuse_paired_validation.resolve(), paired_path)
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
            "--scenario-count",
            str(args.validation_samples),
            "--scenario-seed",
            str(args.validation_seed),
            "--sampling-scheme",
            str(args.validation_sampling_scheme),
            "--oracle-workers",
            str(args.validation_workers),
            "--winner-tolerance-mw",
            str(args.winner_tolerance_mw),
        ]
        if args.validation_sigma is not None:
            validation_command.extend(
                ["--global-lognormal-sigma", str(args.validation_sigma)]
            )
        if args.validation_alpha is not None:
            validation_command.extend(["--cvar-alpha", str(args.validation_alpha)])
        _run_command(
            validation_command,
            cwd=project_root,
            log_path=run_dir / "logs" / "paired_validation.log",
            environment=scheduler_environment,
        )

    comparison_dir = generate_unambiguous_comparison(
        deterministic_snapshot,
        cvar_snapshot,
        paired_path,
        output_dir=run_dir / "visuals" / "comparison",
        show=args.show,
    )
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    verdict = paired["comparison"]

    manifest = {
        "project_root": str(project_root),
        "config": str(config_path),
        "deterministic_results": str(deterministic_snapshot),
        "cvar_results": str(cvar_snapshot),
        "paired_validation": str(paired_path),
        "comparison_visuals": str(comparison_dir),
        "tail_risk_winner": verdict["tail_risk_winner"],
        "conclusion": verdict["conclusion"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("\n=== Completed deterministic + CVaR paired comparison ===")
    print(f"Run directory: {run_dir}")
    print(f"Tail-risk winner: {verdict['tail_risk_winner']}")
    print(verdict["conclusion"])


if __name__ == "__main__":
    main()
