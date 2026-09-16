"""Run the annual paired Monte Carlo N-1 evaluator directly from PyCharm.

Edit only the USER SETTINGS block below, then press Run on this file.
No command-line arguments are required.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make direct execution work from PyCharm or by double-clicking the file.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scheduler_clustered.annual_mc_evaluator import (  # noqa: E402
    AnnualMCConfig,
    evaluate_full_horizon_mc,
    load_schedule_pair,
)
from scheduler_clustered.runtime import (  # noqa: E402
    apply_data_overrides_from_json,
    configured_decomposition_data,
    dataclass_config_from_json,
)


# =============================================================================
# USER SETTINGS — edit these values only
# =============================================================================

# Main PROPER scheduler configuration.
CONFIG_FILE = _REPO_ROOT / "config" / "IEEE24_scheduler_v2.json"

# Choose one input mode:
#   "common"   -> one common_risk_comparison_results.json file
#   "separate" -> deterministic and CVaR result files from separate schedulers
INPUT_MODE = "common"

# Used when INPUT_MODE == "common".
COMMON_RESULTS_FILE = (
    _REPO_ROOT
    / "scheduler_clustered"
    / "outputs"
    / "cluster_scheduler"
    / "common_risk_comparison_results.json"
)

# Used when INPUT_MODE == "separate".
DETERMINISTIC_RESULTS_FILE = (
    _REPO_ROOT
    / "scheduler_clustered"
    / "outputs"
    / "cluster_scheduler"
    / "clustered_deterministic_results.json"
)
CVAR_RESULTS_FILE = (
    _REPO_ROOT
    / "scheduler_clustered"
    / "outputs"
    / "cluster_scheduler"
    / "clustered_cvar_results.json"
)

# Optional MC settings file. Set to None to use AnnualMCConfig defaults below.
MC_CONFIG_FILE = _THIS_FILE.parent / "config_annual_mc_example.json"

# Run controls. These override values in MC_CONFIG_FILE when not None.
MC_YEARS = 20
WORKERS = 1
RANDOM_SEED = 20260731

# Output location. A timestamped run folder is created automatically.
OUTPUT_ROOT = (
    _REPO_ROOT
    / "scheduler_clustered"
    / "outputs"
    / "cluster_scheduler"
    / "mc_evaluation"
)

# =============================================================================


def _require_file(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n  {path}\n"
            "Update the corresponding path in the USER SETTINGS block."
        )
    return path


def main() -> None:
    config_file = _require_file(CONFIG_FILE, "Scheduler configuration")

    mode = INPUT_MODE.strip().lower()
    if mode == "common":
        primary_results = _require_file(COMMON_RESULTS_FILE, "Common comparison result")
        cvar_results = None
    elif mode == "separate":
        primary_results = _require_file(
            DETERMINISTIC_RESULTS_FILE, "Deterministic scheduler result"
        )
        cvar_results = _require_file(CVAR_RESULTS_FILE, "CVaR scheduler result")
    else:
        raise ValueError("INPUT_MODE must be either 'common' or 'separate'.")

    mc_config_file = None
    if MC_CONFIG_FILE is not None:
        mc_config_file = _require_file(Path(MC_CONFIG_FILE), "Annual MC configuration")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT.resolve() / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                run_dir / "logs" / "annual_mc_evaluation.log", encoding="utf-8"
            ),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    overrides: dict[str, int] = {}
    if MC_YEARS is not None:
        overrides["mc_years"] = int(MC_YEARS)
    if WORKERS is not None:
        overrides["oracle_workers"] = int(WORKERS)
    if RANDOM_SEED is not None:
        overrides["seed"] = int(RANDOM_SEED)

    mc_config = dataclass_config_from_json(
        AnnualMCConfig,
        str(mc_config_file) if mc_config_file else None,
        "annual_mc_evaluation",
        defaults=overrides,
    )

    logger.info("Input mode: %s", mode)
    logger.info("Configuration: %s", config_file)
    logger.info("Primary results: %s", primary_results)
    if cvar_results is not None:
        logger.info("CVaR results: %s", cvar_results)
    logger.info("Output directory: %s", run_dir)

    data = configured_decomposition_data(
        str(config_file), allow_outage_deferral=True
    )
    data = apply_data_overrides_from_json(
        data, str(mc_config_file) if mc_config_file else None
    )
    schedules = load_schedule_pair(
        str(primary_results), str(cvar_results) if cvar_results else None
    )
    summary = evaluate_full_horizon_mc(data, schedules, mc_config, run_dir)

    manifest = {
        "run_id": run_id,
        "input_mode": mode,
        "config": str(config_file),
        "primary_results": str(primary_results),
        "cvar_results": str(cvar_results) if cvar_results else None,
        "mc_config": str(mc_config_file) if mc_config_file else None,
        "output_dir": str(run_dir),
        "summary": summary,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nAnnual Monte Carlo evaluation completed.")
    print(f"Results: {run_dir}")
    print(f"Visuals: {run_dir / 'visuals'}")


if __name__ == "__main__":
    main()
