"""One-click deterministic-versus-CVaR comparison and Monte Carlo validation.

Place this file in ``PROPER/scheduler_clustered/`` and run it directly from
PyCharm. No command-line arguments are required.

Workflow
--------
1. Solve the clustered deterministic scheduler.
2. Solve the clustered CVaR scheduler.
3. Generate formulation-specific and paired-comparison plots.
4. Evaluate both final schedules on the same paired full-horizon Monte Carlo
   load trajectories and full valid N-1 contingency set.
5. Generate Monte Carlo risk plots and save all outputs in one timestamped run.

The Monte Carlo load sampler inherits the Gaussian marginal/spatial parameters
used by the CVaR scheduler: relative sigma, global sigma, correlation
shrinkage, antithetic sampling, CVaR confidence level, and DNS loss weighting.
``MC_TEMPORAL_RHO = 0`` reproduces the CVaR model's absence of temporal error
correlation. Set it above zero only when a temporally correlated annual error
process is intentionally required.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# Direct execution support
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parent
_REPO_ROOT = _PACKAGE_DIR.parent


def _bootstrap_repo_imports() -> None:
    """Prevent ``scheduler_clustered/visualization.py`` shadowing the
    repository-level ``visualization`` package when this file is run directly.

    PyCharm normally places the script directory at ``sys.path[0]``. Because
    this package contains ``scheduler_clustered/visualization.py``, the legacy
    import ``visualization.visualize_schedule`` in ``optimizers.gurobi_SCOS``
    can otherwise resolve to that module instead of the repository package.
    """
    package_dir = _PACKAGE_DIR.resolve()
    repo_root = _REPO_ROOT.resolve()

    # Run with the repository as working directory, as expected by the legacy
    # PROPER data loader and relative configuration paths.
    os.chdir(repo_root)

    cleaned: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except (OSError, RuntimeError):
            cleaned.append(entry)
            continue
        if resolved == package_dir:
            continue
        if resolved == repo_root:
            continue
        cleaned.append(entry)

    sys.path[:] = [str(repo_root), *cleaned]

    # Protect interactive/PyCharm sessions in which the wrong top-level module
    # may already have been imported during a previous execution.
    loaded = sys.modules.get("visualization")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None:
        try:
            if Path(loaded_file).resolve() == package_dir / "visualization.py":
                del sys.modules["visualization"]
        except (OSError, RuntimeError):
            pass


_bootstrap_repo_imports()
os.environ.setdefault("MPLBACKEND", "Agg")

from scheduler_clustered.annual_mc_evaluator import (  # noqa: E402
    AnnualMCConfig,
    evaluate_full_horizon_mc,
    load_schedule_pair,
)
from scheduler_clustered.run_cluster_comparison import main as run_comparison  # noqa: E402
from scheduler_clustered.runtime import (  # noqa: E402
    apply_data_overrides_from_json,
    configured_decomposition_data,
)


# =============================================================================
# USER SETTINGS — edit only this block
# =============================================================================

# Main network, demand, outage and scheduler data configuration.
SCHEDULER_CONFIG = _REPO_ROOT / "config" / "conf_IEEE24_scheduler_v2.json"

# Clustered formulation configurations already present in scheduler_clustered/.
DETERMINISTIC_CONFIG = _PACKAGE_DIR / "config_clustered_example.json"
CVAR_CONFIG = _PACKAGE_DIR / "config_clustered_cvar_example.json"

# Parent directory for all comparison runs.
OUTPUT_ROOT = _REPO_ROOT / "outputs" / "clustered_comparison_runs"

# Comparison validation used immediately after optimisation.
PAIRED_VALIDATION_SAMPLES = 96
PAIRED_VALIDATION_WORKERS = 1
PAIRED_VALIDATION_SEED = 20260727

# Full-horizon paired Monte Carlo validation.
MC_YEARS = 20
MC_WORKERS = 1
MC_SEED = 20260731

# Set to 0.0 to retain the same independent-in-time Gaussian error assumption
# used by the CVaR scheduler. A value such as 0.9 introduces annual persistence.
MC_TEMPORAL_RHO = 0.0

# The current IEEE-24 scheduling case uses daily planning steps. Therefore each
# step contributes 24 h to annual DNS exposure. Change to 1.0 for hourly steps.
MC_TIME_STEP_HOURS = 24.0

# Complete N-1 validation is recommended for final schedule comparison.
MC_USE_ALL_CONTINGENCIES = True
MC_CHECKPOINT_EVERY_YEARS = 1

# Set True only to rerun plotting/MC from existing deterministic and CVaR root
# result files. Normal use should remain False.
REUSE_OPTIMISATION_RESULTS = False

# =============================================================================


def _require_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n  {resolved}\n"
            "Update the corresponding path in the USER SETTINGS block."
        )
    return resolved


def _load_cvar_settings(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    section = document.get("clustered_cvar")
    if not isinstance(section, dict):
        raise ValueError(f"Missing 'clustered_cvar' object in {path}.")
    model = section.get("scenario_model", "gaussian_critical_states")
    if model != "gaussian_critical_states":
        raise ValueError(
            "This runner currently requires the CVaR Gaussian critical-state "
            f"load model, but scenario_model={model!r}."
        )
    return section


def _configure_logging(run_dir: Path) -> logging.Logger:
    log_path = run_dir / "logs" / "main_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("scheduler_clustered.main_run_comparison_mc")


def main() -> None:
    scheduler_config = _require_file(SCHEDULER_CONFIG, "Scheduler configuration")
    deterministic_config = _require_file(
        DETERMINISTIC_CONFIG, "Deterministic clustered configuration"
    )
    cvar_config = _require_file(CVAR_CONFIG, "CVaR clustered configuration")
    cvar_settings = _load_cvar_settings(cvar_config)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _configure_logging(run_dir)

    logger.info("Combined comparison and MC run: %s", run_id)
    logger.info("Output directory: %s", run_dir)

    # ------------------------------------------------------------------
    # 1) Deterministic + CVaR optimisation, paired validation and plots
    # ------------------------------------------------------------------
    comparison_args = [
        "--project-root",
        str(_REPO_ROOT),
        "--config",
        str(scheduler_config),
        "--deterministic-config",
        str(deterministic_config),
        "--cvar-config",
        str(cvar_config),
        "--output-dir",
        str(run_dir),
        "--validation-samples",
        str(PAIRED_VALIDATION_SAMPLES),
        "--validation-scenario-model",
        "gaussian_critical_states",
        "--validation-critical-states",
        str(int(cvar_settings.get("critical_state_count", 3))),
        "--validation-seed",
        str(PAIRED_VALIDATION_SEED),
        "--validation-relative-sigma",
        str(float(cvar_settings.get("gaussian_relative_sigma", 0.05))),
        "--validation-global-sigma",
        str(float(cvar_settings.get("gaussian_global_sigma", 0.03))),
        "--validation-alpha",
        str(float(cvar_settings.get("cvar_alpha", 0.95))),
        "--validation-workers",
        str(PAIRED_VALIDATION_WORKERS),
    ]
    if REUSE_OPTIMISATION_RESULTS:
        comparison_args.append("--reuse-results")

    logger.info("Starting deterministic-versus-CVaR comparison.")
    run_comparison(comparison_args)

    deterministic_results = (
        run_dir / "results" / "clustered_deterministic_results.json"
    )
    cvar_results = run_dir / "results" / "clustered_cvar_results.json"
    _require_file(deterministic_results, "Deterministic comparison result")
    _require_file(cvar_results, "CVaR comparison result")

    # ------------------------------------------------------------------
    # 2) Full-horizon paired MC using the CVaR load-distribution settings
    # ------------------------------------------------------------------
    mc_dir = run_dir / "mc_evaluation"
    (mc_dir / "logs").mkdir(parents=True, exist_ok=True)

    mc_log_handler = logging.FileHandler(
        mc_dir / "logs" / "annual_mc_evaluation.log", encoding="utf-8"
    )
    mc_log_handler.setFormatter(
        logging.Formatter("%(asctime)s::%(levelname)s::%(name)s::%(message)s")
    )
    logging.getLogger().addHandler(mc_log_handler)

    mc_config = AnnualMCConfig(
        mc_years=int(MC_YEARS),
        seed=int(MC_SEED),
        alpha=float(cvar_settings.get("cvar_alpha", 0.95)),
        temporal_rho=float(MC_TEMPORAL_RHO),
        relative_sigma=float(cvar_settings.get("gaussian_relative_sigma", 0.05)),
        global_sigma=float(cvar_settings.get("gaussian_global_sigma", 0.03)),
        correlation_shrinkage=float(
            cvar_settings.get("gaussian_correlation_shrinkage", 0.20)
        ),
        antithetic=bool(cvar_settings.get("gaussian_antithetic", True)),
        time_step_hours=float(MC_TIME_STEP_HOURS),
        oracle_workers=int(MC_WORKERS),
        total_dns_component_weight=float(
            cvar_settings.get("total_dns_component_weight", 0.10)
        ),
        use_all_contingencies=bool(MC_USE_ALL_CONTINGENCIES),
        checkpoint_every_years=int(MC_CHECKPOINT_EVERY_YEARS),
    )

    logger.info("Starting paired full-horizon Monte Carlo validation.")
    logger.info("MC configuration: %s", asdict(mc_config))

    data = configured_decomposition_data(
        str(scheduler_config), allow_outage_deferral=True
    )
    # Apply the same data overrides used by the CVaR scheduler.
    data = apply_data_overrides_from_json(data, str(cvar_config))

    schedules = load_schedule_pair(
        str(deterministic_results), str(cvar_results)
    )
    mc_summary = evaluate_full_horizon_mc(data, schedules, mc_config, mc_dir)

    # ------------------------------------------------------------------
    # 3) Consolidated manifest
    # ------------------------------------------------------------------
    comparison_manifest_path = run_dir / "run_manifest.json"
    comparison_manifest = {}
    if comparison_manifest_path.is_file():
        comparison_manifest = json.loads(
            comparison_manifest_path.read_text(encoding="utf-8")
        )

    manifest = {
        **comparison_manifest,
        "run_id": run_id,
        "combined_workflow": "deterministic_vs_cvar_plus_full_horizon_mc",
        "scheduler_config": str(scheduler_config),
        "deterministic_config": str(deterministic_config),
        "cvar_config": str(cvar_config),
        "comparison_results": {
            "deterministic": str(deterministic_results),
            "cvar": str(cvar_results),
            "paired_validation": str(
                run_dir / "results" / "paired_risk_validation.json"
            ),
        },
        "mc_distribution_consistency": {
            "scenario_model": cvar_settings.get("scenario_model"),
            "relative_sigma": mc_config.relative_sigma,
            "global_sigma": mc_config.global_sigma,
            "correlation_shrinkage": mc_config.correlation_shrinkage,
            "antithetic": mc_config.antithetic,
            "alpha": mc_config.alpha,
            "total_dns_component_weight": mc_config.total_dns_component_weight,
            "temporal_rho": mc_config.temporal_rho,
            "note": (
                "Marginal and spatial Gaussian parameters are inherited from "
                "the CVaR scheduler. Temporal correlation is controlled "
                "separately for annual trajectories."
            ),
        },
        "mc_output_dir": str(mc_dir),
        "mc_summary": mc_summary,
    }
    comparison_manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    logger.info("Combined workflow completed successfully.")
    print("\n=== COMPLETE ===")
    print(f"Run directory:       {run_dir}")
    print(f"Comparison plots:    {run_dir / 'visuals' / 'comparison'}")
    print(f"Monte Carlo results: {mc_dir / 'results'}")
    print(f"Monte Carlo plots:   {mc_dir / 'visuals'}")


if __name__ == "__main__":
    main()