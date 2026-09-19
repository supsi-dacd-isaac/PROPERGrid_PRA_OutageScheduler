"""Run compact static-PRA and ROSPRA studies on common power-system cases.

Examples
--------
Quick public-case smoke run::

    python demos/run_pra_comparison.py --cases ieee24 ieee118 --samples 10 --max-contingencies 12

Complete two-hour comparison, including the protected Swissgrid case::

    python demos/run_pra_comparison.py --cases ieee24 ieee118 swissgrid --hours 2 --samples 50

Weather-conditioned extension::

    python demos/run_pra_comparison.py --weather --weather-csv data/weather/hourly_wind.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pra_psa.ROSPRA.network import DCNetwork
from pra_psa.ROSPRA.risk import CascadingFailureSimulator, Contingency as ROSContingency, PRAEngine
from pra_psa.core.contingencies import build_n1_contingencies, build_nk_contingencies
from pra_psa.rospra_time_series import pandapower_profiles_to_bus, run_time_series_rospra
from pra_psa.time_series_pra import (
    GaussianRelativeLoadSampler,
    PRAConfig,
    UniformProbabilityModel,
    WeatherConditionedPoissonModel,
    run_time_series_pra,
)
from probabilistic_models.contingencies.contingencies import ComponentSpec, ComponentType, PoissonFailureModel
from probabilistic_models.weather.weather import EmpiricalWeatherModel, PlaceholderWeatherModel
from visualization.pra_comparison import plot_case_summary, plot_time_varying_risk


LOGGER = logging.getLogger("pra_comparison")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("ieee24", "ieee118", "swissgrid"),
        default=("ieee24", "ieee118", "swissgrid"),
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("contingency", "rospra"),
        default=("contingency", "rospra"),
    )
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pf-mode", choices=("ac", "dc"), default="ac")
    parser.add_argument("--reference-dispatch", choices=("opf", "pf"), default="opf")
    parser.add_argument("--max-contingencies", type=int, default=None)
    parser.add_argument(
        "--n2-contingencies",
        type=int,
        default=0,
        metavar="COUNT",
        help="Append COUNT randomly selected N-2 branch/transformer contingencies.",
    )
    parser.add_argument("--total-contingency-probability", type=float, default=0.01)
    parser.add_argument("--common-load-sigma", type=float, default=0.035)
    parser.add_argument("--nodal-load-sigma", type=float, default=0.015)
    parser.add_argument("--weather", action="store_true")
    parser.add_argument("--weather-csv", type=Path, default=None)
    parser.add_argument("--weather-regime-column", default=None)
    parser.add_argument("--weather-wind-column", default="wind_speed_m_s")
    parser.add_argument("--load-hazard-beta", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "pra_comparison",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of recording a skipped case when protected Swissgrid inputs are absent.",
    )
    args = parser.parse_args(argv)
    if args.hours < 1 or args.start_hour < 0:
        parser.error("--hours must be positive and --start-hour non-negative.")
    if args.samples < 2:
        parser.error("--samples must be at least 2.")
    if args.max_contingencies is not None and args.max_contingencies < 1:
        parser.error("--max-contingencies must be positive.")
    if args.n2_contingencies < 0:
        parser.error("--n2-contingencies cannot be negative.")
    return args


def _load_swissgrid_case() -> Any:
    case_path = ROOT / "data" / "powersystems" / "Swiss2025" / "case_swissgrid2025.py"
    if not case_path.exists():
        raise FileNotFoundError(
            "The NDA-protected Swissgrid builder is absent: " + str(case_path)
        )
    spec = importlib.util.spec_from_file_location("proper_swissgrid_case", case_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.case_swissgrid2025()


def load_case(case_name: str) -> Any:
    if case_name == "swissgrid":
        return _load_swissgrid_case()
    import pandapower.networks as networks

    if case_name == "ieee24":
        return networks.case24_ieee_rts()
    if case_name == "ieee118":
        return networks.case118()
    raise ValueError(case_name)


def _to_load_profiles(net: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Map a load-level or bus-level table to one column per net.load."""

    values = data.apply(pd.to_numeric, errors="raise")
    if values.shape[1] == len(net.load):
        values.columns = net.load.index
        return values
    if values.shape[1] != len(net.bus):
        raise ValueError(
            f"Profile table has {values.shape[1]} columns; expected {len(net.load)} loads "
            f"or {len(net.bus)} buses."
        )
    output = pd.DataFrame(0.0, index=values.index, columns=net.load.index)
    nominal = pd.to_numeric(net.load["p_mw"], errors="coerce").fillna(0.0)
    for bus_position, bus_index in enumerate(net.bus.index):
        load_indices = net.load.index[net.load["bus"] == bus_index]
        if len(load_indices) == 0:
            continue
        weights = nominal.loc[load_indices].abs().to_numpy(dtype=float)
        if weights.sum() <= 1e-12:
            weights = np.full(len(load_indices), 1.0 / len(load_indices))
        else:
            weights /= weights.sum()
        output.loc[:, load_indices] = values.iloc[:, bus_position].to_numpy()[:, None] * weights
    return output


def load_profiles(case_name: str, net: Any, *, start_hour: int, hours: int) -> tuple[pd.DataFrame, str]:
    from pra_psa.data import load_hourly_demand

    directory_names = {
        "ieee24": "IEEE24",
        "ieee118": "IEEE118",
        "swissgrid": "Swiss2025",
    }
    system_dir = ROOT / "data" / "powersystems" / directory_names[case_name]
    demand = load_hourly_demand(system_dir) if system_dir.exists() else None
    if demand is not None and not demand.empty:
        stop = start_hour + hours
        if stop > len(demand):
            raise ValueError(f"{case_name}: requested profile rows {start_hour}:{stop}, only {len(demand)} exist.")
        profiles = _to_load_profiles(net, demand.iloc[start_hour:stop].copy())
        profiles.index = pd.RangeIndex(start_hour, stop, name="time")
        return profiles, f"measured/file-based: {system_dir}"

    # Reproducible fallback for public benchmark cases only. It is labelled in
    # metadata and must not be interpreted as historical operating data.
    if case_name == "swissgrid":
        raise FileNotFoundError(
            f"No hourly Swissgrid load profiles were found under {system_dir}."
        )
    nominal = pd.to_numeric(net.load["p_mw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hour = np.arange(start_hour, start_hour + hours)
    scale = 1.0 + 0.08 * np.sin(2.0 * np.pi * (hour - 8.0) / 24.0)
    profiles = pd.DataFrame(
        scale[:, None] * nominal[None, :],
        index=pd.RangeIndex(start_hour, start_hour + hours, name="time"),
        columns=net.load.index,
    )
    return profiles, "synthetic benchmark scaling (no historical profile file found)"


def build_contingencies(net: Any, args: argparse.Namespace) -> list[Any]:
    contingencies = sorted(
        build_n1_contingencies(net, include=("line", "trafo")),
        key=lambda item: item.contingency_id,
    )
    if args.max_contingencies is not None:
        contingencies = contingencies[: args.max_contingencies]
    if args.n2_contingencies:
        contingencies.extend(
            build_nk_contingencies(
                net,
                k=2,
                include=("line", "trafo"),
                max_contingencies=args.n2_contingencies,
                random_state=args.seed,
            )
        )
    return contingencies


def make_weather_and_probability(args: argparse.Namespace) -> tuple[Any | None, Any]:
    if not args.weather:
        return None, UniformProbabilityModel(args.total_contingency_probability)
    if args.weather_csv is None:
        weather = PlaceholderWeatherModel()
    else:
        weather_data = pd.read_csv(args.weather_csv)
        weather = EmpiricalWeatherModel.fit(
            weather_data,
            regime_column=args.weather_regime_column,
            wind_speed_column=args.weather_wind_column,
        )
    probabilities = WeatherConditionedPoissonModel(
        rates_per_hour={"line": 2.26e-7 * 50.0, "trafo": 6.27e-6, "default": 1e-6},
        weather_multipliers={"H1_adverse": 3.0, "H2_normal": 1.0},
        load_beta=args.load_hazard_beta,
    )
    return weather, probabilities


def make_rospra_engine(
    net: Any,
    weather_model: Any | None,
    contingencies: list[Any],
) -> PRAEngine:
    network = DCNetwork.from_pandapower(net)
    components = [
        ComponentSpec(
            branch.asset_id,
            ComponentType(branch.component_kind),
            length_km=branch.length_km,
            metadata=branch.metadata,
        )
        for branch in network.branches
    ]
    failure_model = PoissonFailureModel(components, default_line_length_km=50.0)
    simulator = CascadingFailureSimulator(network, trip_policy="most_overloaded")
    rospra_contingencies = [
        ROSContingency(
            item.contingency_id,
            tuple(f"{outage.element_type}:{outage.element_index}" for outage in item.outages),
            description="shared static-PRA/ROSPRA contingency catalogue",
        )
        for item in contingencies
    ]
    return PRAEngine(
        simulator,
        failure_model,
        weather_model=weather_model or PlaceholderWeatherModel(),
        contingencies=rospra_contingencies,
    )


def run_case(case_name: str, args: argparse.Namespace) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    LOGGER.info("Loading %s", case_name)
    net = load_case(case_name)
    profiles, profile_source = load_profiles(case_name, net, start_hour=args.start_hour, hours=args.hours )
    contingencies = build_contingencies(net, args)
    sampler = GaussianRelativeLoadSampler( common_sigma=args.common_load_sigma,nodal_sigma=args.nodal_load_sigma,)
    weather, probability_model = make_weather_and_probability(args)
    case_dir = args.output_dir / case_name
    time_frames: list[pd.DataFrame] = []
    case_summary: dict[str, Any] = {
        "case": case_name,         "status": "completed",
        "n_buses": int(len(net.bus)),  "n_lines": int(len(net.line)),
        "n_transformers": int(len(net.trafo)), "n_contingencies_static": int(len(contingencies)),
        "profile_source": profile_source,     }

    if "contingency" in args.engines:
        LOGGER.info("%s: static PRA (%d contingencies, %d samples, %d times)", case_name, len(contingencies), args.samples, len(profiles))
        result = run_time_series_pra(net, profiles, contingencies,
                                     config=PRAConfig(n_samples=args.samples, pf_mode=args.pf_mode,
                                                      reference_dispatch=args.reference_dispatch,
                                                      seed=args.seed,
                                                      ),
                                     operating_state_sampler=sampler,
                                     probability_model=probability_model,
                                     weather_model=weather,)
        result.save(case_dir / "contingency")
        time_frame = result.risk_by_time.copy()
        time_frame.insert(0, "case", case_name)
        time_frames.append(time_frame)
        case_summary.update(
            {
                "mean_composite_risk_per_100mw": float(time_frame["risk_composite_per_100mw"].mean()),
                "maximum_worst_performance_score": result.summary["maximum_worst_performance_score"],
                "static_failed_power_flows": result.summary["failed_power_flow_count"],
            }
        )

    if "rospra" in args.engines:
        LOGGER.info("%s: ROSPRA cascading-risk study", case_name)
        engine = make_rospra_engine(net, weather, contingencies)
        bus_profiles = pandapower_profiles_to_bus(net, profiles)
        rospra = run_time_series_rospra( engine, bus_profiles,
                                         n_samples=args.samples,  sampler=sampler,  seed=args.seed, )
        rospra.save(case_dir / "rospra")
        time_frame = rospra.risk_by_time.copy()
        time_frame.insert(0, "case", case_name)
        time_frames.append(time_frame)
        case_summary.update(
            {
                "mean_expected_dns_per_100mw": float(time_frame["expected_dns_per_100mw"].mean()),
                "maximum_dns_mw": rospra.summary["maximum_dns_mw"],
            }
        )
    return time_frames, case_summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    time_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for case_name in args.cases:
        try:
            frames, summary = run_case(case_name, args)
            time_frames.extend(frames)
            summaries.append(summary)
        except (FileNotFoundError, ImportError) as exc:
            if args.strict or case_name != "swissgrid":
                raise
            LOGGER.warning("Skipping %s: %s", case_name, exc)
            summaries.append({"case": case_name, "status": "skipped", "reason": str(exc)})

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "case_comparison.csv", index=False)
    if time_frames:
        stacked = pd.concat(time_frames, ignore_index=True, sort=False)
        combined = stacked.groupby(["case", "time"], as_index=False, sort=False).first()
        combined.to_csv(args.output_dir / "time_varying_risk.csv", index=False)
        plot_time_varying_risk(combined, args.output_dir / "time_varying_risk.png")
    completed = summary.loc[summary["status"] == "completed"].copy()
    if not completed.empty and (
        "mean_composite_risk_per_100mw" in completed
        or "mean_expected_dns_per_100mw" in completed
    ):
        plot_case_summary(completed, args.output_dir / "case_comparison.png")

    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "notes": [
            "Static-PRA and ROSPRA values have different consequence definitions and are not added together.",
            "Synthetic benchmark profiles are explicitly labelled in case_comparison.csv.",
            "A skipped Swissgrid row means the NDA-protected case/profile files were unavailable.",
        ],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
