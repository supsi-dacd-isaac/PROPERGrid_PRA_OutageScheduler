"""Single-formulation command line runner and compatibility utilities."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from scheduler_monolitic.config import ModelConfig, OutputConfig, RiskConfig, ScenarioConfig, SolverConfig
from scheduler_monolitic.data_adapter import load_data_from_conf_grid_case
from scheduler_monolitic.postprocessing import save_result_bundle
from scheduler_monolitic.scenario_generation import generate_demand_scenarios


def get_gurobi_solution(model: Any, save_res_dir: str | Path | None = None) -> dict[str, float] | None:
    """Return and optionally save the current Gurobi solution."""
    if getattr(model, "SolCount", 0) < 1:
        return None
    solution = {variable.VarName: float(variable.X) for variable in model.getVars()}
    if save_res_dir is not None:
        path = Path(save_res_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(solution, stream)
    return solution


def get_and_save_solution(
    model: Any,
    save_res_dir: str | Path | None = None,
    case_name: str | None = None,
    aggregation_time: str | None = None,
) -> dict[str, float] | None:
    if save_res_dir is None:
        name = f"monolithic_{case_name or 'case'}_{aggregation_time or 'step'}.json"
        save_res_dir = OutputConfig().resolve_root(__file__) / name
    return get_gurobi_solution(model, save_res_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve one monolithic SCOS formulation.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--formulation", choices=["expected", "cvar", "weighted"], default="expected")
    parser.add_argument("--aggregation-time", type=str, default=None)
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--contingency-top-k", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-estimated-variables", type=int, default=5_000_000)
    parser.add_argument("--allow-large-model", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from scheduler_monolitic.monolithic_scos import solve_scos

    data = load_data_from_conf_grid_case(
        args.config,
        aggregation_time_override=args.aggregation_time,
    )
    bank = generate_demand_scenarios(
        data["nodal_demand"],
        ScenarioConfig(n_scenarios=args.scenarios, seed=args.seed),
    )
    result = solve_scos(
        data,
        bank,
        risk_config=RiskConfig(beta=args.beta, formulation=args.formulation),
        model_config=ModelConfig(
            contingency_top_k=args.contingency_top_k,
            max_estimated_variables=args.max_estimated_variables,
            allow_large_model=args.allow_large_model,
        ),
        solver_config=SolverConfig(time_limit=args.time_limit, threads=args.threads, seed=args.seed),
    )
    root = args.output_dir or OutputConfig().resolve_root(__file__)
    run_name = f"{args.formulation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    directory = save_result_bundle(result, root, run_name)
    print(f"Results saved in: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
