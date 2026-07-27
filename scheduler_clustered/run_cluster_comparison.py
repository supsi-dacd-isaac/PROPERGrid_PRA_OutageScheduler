"""Run deterministic and CVaR clustered schedulers and compare their outputs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    try:
        sys.path.remove(project_root_str)
    except ValueError:
        pass
    sys.path.insert(0, project_root_str)

os.environ.setdefault("MPLBACKEND", "Agg")

from scheduler_clustered.comparison_visualization import generate_comparison_visuals
from scheduler_clustered.results_visualization import generate_all_plots


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_module(module: str, *, cwd: Path, log_path: Path) -> None:
    command = [sys.executable, "-m", module]
    print(f"\n=== Running {' '.join(command)} ===")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(cwd), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{module} failed with exit code {return_code}. See {log_path}.")


def _fresh_result(path: Path, *, module: str, project_root: Path, log_path: Path) -> Path:
    if path.exists():
        path.unlink()
    _run_module(module, cwd=project_root, log_path=log_path)
    if not path.exists():
        raise FileNotFoundError(f"{module} completed without creating {path}.")
    return path


def _copy_result(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _resolve_reuse_path(explicit: Path | None, project_root: Path, filename: str) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit if explicit.is_absolute() else project_root / explicit)
    candidates.extend(
        [
            project_root / filename,
            project_root / "scheduler_clustered" / filename,
            project_root / "example_results" / filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate {filename}. Checked: {candidates}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=_default_project_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--deterministic-results", type=Path, default=None)
    parser.add_argument("--cvar-results", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--deterministic-module", default="scheduler_clustered.main_clustered")
    parser.add_argument("--cvar-module", default="scheduler_clustered.main_clustered_cvar")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    package_dir = project_root / "scheduler_clustered"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"scheduler_clustered package not found under {project_root}.")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = project_root / "clustered_comparison_runs" / f"run_{stamp}"
    else:
        run_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_results:
        det_source = _resolve_reuse_path(args.deterministic_results, project_root, "outputs/cluster_scheduler/clustered_deterministic_results.json")
        cvar_source = _resolve_reuse_path(args.cvar_results, project_root, "outputs/cluster_scheduler/clustered_cvar_results.json")
    else:
        det_source = _fresh_result(
            project_root / "clustered_deterministic_results.json",
            module=args.deterministic_module,
            project_root=project_root,
            log_path=run_dir / "logs" / "deterministic.log",
        )
        cvar_source = _fresh_result(
            project_root / "clustered_cvar_results.json",
            module=args.cvar_module,
            project_root=project_root,
            log_path=run_dir / "logs" / "cvar.log",
        )

    det_snapshot = _copy_result(det_source, run_dir / "results" / "clustered_deterministic_results.json")
    cvar_snapshot = _copy_result(cvar_source, run_dir / "results" / "clustered_cvar_results.json")

    det_visuals = generate_all_plots(det_snapshot, output_dir=run_dir / "visuals" / "deterministic", show=args.show)
    cvar_visuals = generate_all_plots(cvar_snapshot, output_dir=run_dir / "visuals" / "cvar", show=args.show)
    comparison_visuals = generate_comparison_visuals(
        det_snapshot,
        cvar_snapshot,
        output_dir=run_dir / "visuals" / "comparison",
        show=args.show,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "project_root": str(project_root),
        "reuse_results": bool(args.reuse_results),
        "deterministic_source": str(det_source),
        "cvar_source": str(cvar_source),
        "deterministic_snapshot": str(det_snapshot),
        "cvar_snapshot": str(cvar_snapshot),
        "deterministic_visuals": str(det_visuals),
        "cvar_visuals": str(cvar_visuals),
        "comparison_visuals": str(comparison_visuals),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Completed deterministic + CVaR comparison ===")
    print(f"Run directory: {run_dir}")
    print(f"Deterministic visuals: {det_visuals}")
    print(f"CVaR visuals: {cvar_visuals}")
    print(f"Comparison visuals: {comparison_visuals}")


if __name__ == "__main__":
    main()
