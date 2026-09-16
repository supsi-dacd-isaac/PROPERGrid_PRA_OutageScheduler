"""CLI for paired full-horizon Monte Carlo N-1 schedule validation."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from scheduler_clustered.annual_mc_evaluator import AnnualMCConfig, evaluate_full_horizon_mc, load_schedule_pair
from scheduler_clustered.runtime import apply_data_overrides_from_json, configured_decomposition_data, dataclass_config_from_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PROPER scheduler configuration JSON.")
    parser.add_argument("--results", required=True, help="Common comparison JSON, or deterministic result JSON.")
    parser.add_argument("--cvar-results", default=None, help="Optional separate CVaR result JSON.")
    parser.add_argument("--mc-config", default=None, help="Optional JSON containing annual_mc_evaluation settings.")
    parser.add_argument("--output-root", default="scheduler_clustered/outputs/cluster_scheduler/mc_evaluation")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--years", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "logs" / "annual_mc_evaluation.log", encoding="utf-8"),
        ],
    )

    overrides = {}
    if args.years is not None: overrides["mc_years"] = args.years
    if args.workers is not None: overrides["oracle_workers"] = args.workers
    if args.seed is not None: overrides["seed"] = args.seed
    mc_config = dataclass_config_from_json(
        AnnualMCConfig, args.mc_config, "annual_mc_evaluation", defaults=overrides
    )

    data = configured_decomposition_data(args.config, allow_outage_deferral=True)
    data = apply_data_overrides_from_json(data, args.mc_config)
    schedules = load_schedule_pair(args.results, args.cvar_results)
    summary = evaluate_full_horizon_mc(data, schedules, mc_config, run_dir)

    manifest = {
        "run_id": run_id,
        "config": str(Path(args.config).resolve()),
        "results": str(Path(args.results).resolve()),
        "cvar_results": str(Path(args.cvar_results).resolve()) if args.cvar_results else None,
        "output_dir": str(run_dir.resolve()),
        "summary": summary,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.getLogger(__name__).info("Annual MC evaluation written to %s", run_dir)


if __name__ == "__main__":
    main()
