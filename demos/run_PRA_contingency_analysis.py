#!/usr/bin/env python3
r"""Run nodal Monte Carlo PRA and cascading N-1/N-k analysis on a PROPER system.

Example from the PROPER repository root on Windows:

    python demos/run_PRA_contingency_analysis.py --data-dir "C:\Users\roberto.rocchetta\Documents\GitHub\PROPER\data\powersystems\IEEE24" --system IEEE24 --samples 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probabilistic_models.contingencies.contingencies import (
    ComponentSpec,
    ComponentType,
    EmpiricalWeatherModel,
    GaussianCopulaModel,
    PlaceholderWeatherModel,
    PoissonFailureModel,
    diagnose_nodal_data,
    load_nodal_data,
)
from pra_psa.ROSPRA import CascadingFailureSimulator, Contingency, DCNetwork, PRAEngine, build_n1_branch_contingencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--case-name", default="IEEE118")
    parser.add_argument("--nodal-file", type=Path, default=None)
    parser.add_argument("--nodal-variable", default=None)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--load-multiplier", type=float, default=1.0)
    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--training-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--planned-outage", action="append", default=[])
    parser.add_argument( "--exceptional",  action="append", default=[], metavar="ASSET1,ASSET2",
                         help="Add an exceptional N-k branch set. Repeat the argument for multiple sets.")
    parser.add_argument("--weather-csv", type=Path, default=None)
    parser.add_argument("--weather-regime-column", default=None)
    parser.add_argument("--weather-wind-column", default="wind_speed_m_s")
    parser.add_argument("--default-line-length-km", type=float, default=None)
    parser.add_argument("--list-branches", action="store_true", help="Print converted branch ids and exit.")

    args = parser.parse_args()
    args.system = args.case_name

    args.data_dir = ROOT / "data" / "powersystems" / args.case_name
    args.output = ROOT / "outputs" / args.case_name
    return args


def _load_proper_network(data_dir: Path, system_name: str) -> DCNetwork:
    try:
        from pra_psa.data import load_system
    except ImportError as error:
        raise RuntimeError(
            "The existing PROPER package 'pra_psa' is required to load the network topology. "
            "Install this revision in the PROPER repository/environment, or run demos/run_rospra_demo.py "
            "for the self-contained verification case."
        ) from error
    loader_root = data_dir.parent if data_dir.name.casefold() == system_name.casefold() else data_dir
    system = load_system(loader_root, system_name, as_pandapower=True)
    return DCNetwork.from_pandapower(system.net)


def main() -> None:
    args = parse_args()

    ## LOAD POWER GRID DATA - TOPOLOGY AND NODAL DEMAND
    network = _load_proper_network(args.data_dir, args.system)
    if args.list_branches:
        branch_table = pd.DataFrame(
            [
                {
                    "asset_id": item.asset_id,
                    "from_bus": item.from_bus,
                    "to_bus": item.to_bus,
                    "kind": item.component_kind,
                    "length_km": item.length_km,
                    "rating_mw": item.rating_mw,
                }
                for item in network.branches
            ]
        )
        print(branch_table.to_string(index=False))
        return
    nodal_source = args.nodal_file or args.data_dir
    dataset = load_nodal_data(
        nodal_source,
        variable=args.nodal_variable,
        expected_nodes=network.n_buses,
    )

    ## PROBABILISTIC LOAD MODEL - SIMPLE COPULA EXAMPLE
    diagnostics = diagnose_nodal_data(dataset.values)
    split = int(args.training_fraction * dataset.values.shape[0])
    if split < 50 or split >= dataset.values.shape[0]:
        raise ValueError("training_fraction leaves insufficient training or validation observations.")
    nodal_model = GaussianCopulaModel().fit(dataset.values[:split])


    ## CONTINGENCIES AND PROBABILISTIC FAILURE MODEL
    components = [
        ComponentSpec(
            branch.asset_id,
            ComponentType(branch.component_kind),
            length_km=branch.length_km,
            metadata=branch.metadata,
        )
        for branch in network.branches
    ]
    failure_model = PoissonFailureModel(
        components, default_line_length_km=args.default_line_length_km
    )
    if args.weather_csv is None:
        weather_model = PlaceholderWeatherModel()
    else:
        weather = pd.read_csv(args.weather_csv)
        weather_model = EmpiricalWeatherModel.fit(
            weather,
            regime_column=args.weather_regime_column,
            wind_speed_column=args.weather_wind_column,
        )

    ## SIMULATION MODEL FOR SEVERITY ASSESSMENT
    simulator = CascadingFailureSimulator(network, trip_policy="most_overloaded")

    contingencies = build_n1_branch_contingencies(simulator)
    for index, specification in enumerate(args.exceptional, start=1):
        assets = tuple(item.strip() for item in specification.split(",") if item.strip())
        if len(assets) < 2:
            raise ValueError("--exceptional must contain at least two comma-separated branch ids.")
        contingencies.append( Contingency(f"exceptional_{index}", assets, description="user-specified exceptional N-k event"))


    ## RUN PRA ENGINE
    engine = PRAEngine(
        simulator,
        failure_model,
        nodal_model=nodal_model,
        weather_model=weather_model,
        contingencies=contingencies,
    )
    result = engine.run(
        n_samples=args.samples,
        horizon_hours=args.horizon_hours,
        planned_outages=args.planned_outage,
        random_state=args.seed,
    )
    result.save(args.output)

    ## SAVE AND PRINT RESULTS
    print(f"System: {args.system}; {network.n_buses} buses; {len(network.branches)} branches")
    print(f"Nodal data: {dataset.values.shape[0]} observations x {dataset.values.shape[1]} nodes")
    print(f"Nodal effective rank: {diagnostics.effective_rank}/{diagnostics.n_active_nodes} active nodes")
    if diagnostics.warning:
        print("DATA WARNING:", diagnostics.warning)
    print("Weather model:", weather_model.metadata.status)
    for warning in failure_model.assumption_warnings:
        print("RATE WARNING:", warning)
    print(f"Expected DNS: {result.summary.expected_dns_mwh:.6g} MWh")
    print(f"Probability of positive DNS: {result.summary.probability_of_positive_dns:.6g}")
    print("\nHighest risk contributions")
    print(result.risk_by_contingency.head(15).to_string(index=False))
    print(f"\nResults written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
