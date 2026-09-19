"""Run a common-bank risk-neutral versus CVaR monolithic SCOS comparison."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:  # Support ``python optimizers/run_comparison_optimizer.py``.
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from scheduler_monolitic.comparison_visualization import plot_comparison_suite
from scheduler_monolitic.config import ModelConfig, OutputConfig, RiskConfig, ScenarioConfig, SolverConfig
from scheduler_monolitic.data_adapter import load_data_from_conf_grid_case
from scheduler_monolitic.postprocessing import save_result_bundle
from scheduler_monolitic.schedule_validation import validate_schedules
from scheduler_monolitic.scenario_generation import generate_demand_scenarios
from scheduler_monolitic.model_size import estimate_extensive_form_size

LOGGER = logging.getLogger("scheduler_monolitic.comparison")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser( description=(
            "Solve risk-neutral and CVaR-informed monolithic outage schedules on a common "
            "training scenario bank, then compare them on an independent paired validation bank." ))
    parser.add_argument("--config", type=Path, default='None', help="Path to the grid scheduler JSON configuration.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Run-output root. Defaults to the repository outputs folder.")
    parser.add_argument("--aggregation-time",  type=str,    default=None,   help=( "Override config aggregation_time before model construction. "
               "Use W for a tractable full-horizon monolithic comparison."  ), )
    parser.add_argument("--training-scenarios", type=int, default=20)
    parser.add_argument("--validation-scenarios", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=20260729)
    parser.add_argument("--sampling-scheme", choices=["random", "stratified_tail"], default="stratified_tail")
    parser.add_argument("--sigma-global", type=float, default=0.05)
    parser.add_argument("--sigma-local", type=float, default=0.02)
    parser.add_argument("--temporal-correlation", type=float, default=0.80)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--cvar-formulation", choices=["cvar", "weighted"], default="cvar")
    parser.add_argument("--expected-weight", type=float, default=1.0)
    parser.add_argument("--cvar-weight", type=float, default=10.0)
    parser.add_argument("--utility-tolerance", type=float, default=0.0)
    parser.add_argument("--contingency-top-k", type=int, default=None)
    parser.add_argument("--validation-contingency-top-k", type=int, default=None)
    parser.add_argument("--redispatch-fraction", type=float, default=0.20)
    parser.add_argument("--no-dc-power-flow", action="store_true")
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quiet-gurobi", action="store_true")
    parser.add_argument(
        "--max-estimated-variables",
        type=int,
        default=5_000_000,
        help="Safety limit applied to the preflight extensive-form variable estimate.",
    )
    parser.add_argument(
        "--allow-large-model",
        action="store_true",
        help="Bypass the monolithic model-size safety guard (not recommended).",
    )
    parser.add_argument(
        "--name-operational-constraints",
        action="store_true",
        help="Store names for millions of operational constraints; increases build time and RAM.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--validation-batch-size", type=int, default=10)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print the extensive-form size estimate and exit before creating a Gurobi model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data = load_data_from_conf_grid_case(
        args.config,
        aggregation_time_override=args.aggregation_time,
    )
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else OutputConfig().resolve_root(__file__)
    )
    run_directory = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    figures_directory = output_root / "figures"
    run_directory.mkdir(parents=True, exist_ok=True)

    training_scenario_config = ScenarioConfig(
        n_scenarios=args.training_scenarios,
        seed=args.training_seed,
        sampling_scheme=args.sampling_scheme,
        sigma_global=args.sigma_global,
        sigma_local=args.sigma_local,
        temporal_correlation=args.temporal_correlation,
    )
    validation_scenario_config = replace(
        training_scenario_config,
        n_scenarios=args.validation_scenarios,
        seed=args.validation_seed,
        sampling_scheme="random",
    )
    training_bank = generate_demand_scenarios(data["nodal_demand"], training_scenario_config)

    available_contingencies = list(data.get("n_minus1_names") or [])
    contingency_count = len(available_contingencies)
    if args.contingency_top_k is not None:
        contingency_count = min(contingency_count, args.contingency_top_k)
    preflight = estimate_extensive_form_size(
        n_times=len(data["nodal_demand"]),
        n_buses=int(data["num_buses"]),
        n_lines=int(data["num_branches"]),
        n_generators=len(data["network"].gen),
        n_outages=len(data["outages"]["names"]),
        n_scenarios=args.training_scenarios,
        n_contingencies=contingency_count,
        use_dc_power_flow=not args.no_dc_power_flow,
        include_base_state=True,
        include_cvar_variables=True,
    )
    LOGGER.info("Training-model preflight: %s.", preflight.summary())
    if args.preflight_only:
        print(preflight.summary())
        return 0

    from scheduler_monolitic.monolithic_scos import solve_scos

    solver_config = SolverConfig(
        mip_gap=args.mip_gap,
        time_limit=args.time_limit,
        threads=args.threads,
        output_flag=0 if args.quiet_gurobi else 1,
        seed=args.training_seed,
    )
    model_config = ModelConfig(
        use_dc_power_flow=not args.no_dc_power_flow,
        corrective_redispatch_fraction=args.redispatch_fraction,
        contingency_top_k=args.contingency_top_k,
        max_estimated_variables=args.max_estimated_variables,
        allow_large_model=args.allow_large_model,
        name_operational_constraints=args.name_operational_constraints,
    )

    LOGGER.info("Solving common-bank risk-neutral schedule.")
    reference_risk = RiskConfig(beta=args.beta, formulation="expected")
    reference = solve_scos(
        data,
        training_bank,
        risk_config=reference_risk,
        model_config=model_config,
        solver_config=solver_config,
    )
    save_result_bundle(reference, run_directory, "training_risk_neutral")

    LOGGER.info("Solving CVaR-informed schedule with the risk-neutral utility as a feasibility floor.")
    cvar_risk = RiskConfig(
        beta=args.beta,
        formulation=args.cvar_formulation,
        expected_weight=args.expected_weight,
        cvar_weight=args.cvar_weight,
        utility_tolerance=args.utility_tolerance,
    )
    cvar = solve_scos(
        data,
        training_bank,
        risk_config=cvar_risk,
        model_config=model_config,
        solver_config=solver_config,
        reference_schedule=reference.schedule,
        reference_utility=reference.maintenance_utility,
    )
    save_result_bundle(cvar, run_directory, "training_cvar")

    run_summary: dict[str, object] = {
        "training": {
            "risk_neutral": reference.risk_metrics,
            "cvar": cvar.risk_metrics,
            "risk_neutral_utility": reference.maintenance_utility,
            "cvar_utility": cvar.maintenance_utility,
        },
        "settings": vars(args),
    }
    run_summary["settings"]["config"] = str(args.config) if args.config else None
    run_summary["settings"]["output_dir"] = str(output_root)

    if not args.skip_validation:
        LOGGER.info("Running independent paired out-of-sample validation.")
        validation_bank = generate_demand_scenarios(data["nodal_demand"], validation_scenario_config)
        validation_top_k = (
            args.validation_contingency_top_k
            if args.validation_contingency_top_k is not None
            else args.contingency_top_k
        )
        validation_model_config = replace(model_config, contingency_top_k=validation_top_k)
        validation = validate_schedules(
            data,
            reference.schedule,
            cvar.schedule,
            validation_bank,
            beta=args.beta,
            model_config=validation_model_config,
            solver_config=replace(solver_config, seed=args.validation_seed),
            batch_size=args.validation_batch_size,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.validation_seed,
            output_dir=run_directory,
        )
        plot_comparison_suite(reference, cvar, validation, figures_directory)
        run_summary["validation"] = validation.summary

        delta = validation.summary["cvar_minus_reference"]
        winner = validation.summary["winner_by_cvar"]
        LOGGER.info("Paired validation winner by CVaR: %s; ΔCVaR = %.6f MW-step.", winner, delta)

    with (run_directory / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(run_summary, stream, indent=2, default=str)

    print(f"Results saved in: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
