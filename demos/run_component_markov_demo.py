#!/usr/bin/env python3
"""Inspect the three-state CTMC implied by Table 3 plus explicit placeholders."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probabilistic_models.contingencies.contingencies import (ComponentSpec,  ComponentType,
                                                              PoissonFailureModel,  ThreeStateMarkovModel,
                                                              WeatherRegime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", default="example_line")
    parser.add_argument("--component-type", choices=[item.value for item in ComponentType], default="single_line")
    parser.add_argument("--length-km", type=float, default=100.0)
    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--mean-forced-repair-hours", type=float, default=24.0)
    parser.add_argument("--planned-outages-per-year", type=float, default=1.0)
    parser.add_argument("--mean-planned-duration-hours", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "markov_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    component = ComponentSpec(
        args.asset_id,
        ComponentType(args.component_type),
        length_km=args.length_km if "line" in args.component_type else None,
    )
    failure_model = PoissonFailureModel([component])
    markov = ThreeStateMarkovModel(
        args.asset_id,
        failure_model,
        forced_repair_rate_per_hour=1.0 / args.mean_forced_repair_hours,
        planned_entry_rate_per_hour=args.planned_outages_per_year / (365.0 * 24.0),
        planned_return_rate_per_hour=1.0 / args.mean_planned_duration_hours,
        placeholder_parameters=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for regime in (WeatherRegime.H1_ADVERSE, WeatherRegime.H2_NORMAL):
        table = markov.transition_table(args.horizon_hours, regime)
        table.to_csv(args.output / f"transition_{regime.value}.csv")
        print(f"\n{regime.value}; P({args.horizon_hours:g} h)")
        print(table.to_string(float_format=lambda value: f"{value:.9g}"))
        print("stationary:", markov.stationary_distribution(regime))
    print(
        "\nPLACEHOLDER NOTICE: repair and planned-outage parameters came from CLI defaults; "
        "replace them with fitted event-history values before interpretation."
    )


if __name__ == "__main__":
    main()
