#!/usr/bin/env python3
"""Run a 24-hour PRA/PSA study on the Swissgrid 2025 research case.

The script performs two related tasks:

1. It evaluates the intact system over 24 operating points and reproduces the
   6 x 4 network figure with bus voltage and line-loading colour scales.
2. It performs an N-1 contingency analysis for the same operating points,
   assigns contingency probabilities, and exports hourly and contingency-level
   risk tables.

The default figure uses base-case AC power-flow results, matching the supplied
example. Risk-informed maps can instead show probability-weighted expected or
worst post-contingency quantities.

Run from the repository root:

    python demos/run_swiss2025_24h_risk_assessment.py

Typical EuA run:

    python demos/run_swiss2025_24h_risk_assessment.py --profile-source eua  --p-file data/powersystems/Swiss2025/Scenarios/20250425_P_EuA_20250423161946.txt --q-file data/powersystems/Swiss2025/Scenarios/20250425_Q_EuA_20250423161946.txt --max-contingencies 20

Expected post-contingency map using AC contingency analysis:

    python demos/run_swiss2025_24h_risk_assessment.py \
        --risk-pf-mode ac \
        --map-mode expected

Fast smoke test:

    python demos/run_swiss2025_24h_risk_assessment.py \
        --max-contingencies 20

Notes
-----
- The Swissgrid case and associated data are NDA-protected and are therefore
  expected in the local repository under ``data/powersystems/Swiss2025``.
- The public repository contains the case-independent PRA/PSA and plotting code,
  but not the protected Swissgrid input data.
- By default, the risk assessment concerns the intact system: the base case and
  credible N-1 contingencies are evaluated with no planned maintenance outage.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pandapower as pp


# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.powersystems.Swiss2025.case_swissgrid2025 import case_swissgrid2025
from pra_psa.core.contingency_analysis import run_contingency_analysis
from pra_psa.core.contingencies import build_n1_contingencies
from pra_psa.data import load_hourly_demand
from pra_psa.risk import UniformContingencyModel, aggregate_risk


LOGGER = logging.getLogger("swiss2025_24h_risk")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HourlyMapState:
    """Values plotted for one operating hour."""

    hour: int
    line_loading_pct: pd.Series
    bus_voltage_pu: pd.Series
    converged: bool
    pf_mode_used: str


@dataclass(frozen=True)
class PreparedProfiles:
    """Load-level active/reactive operating profiles aligned with net.load."""

    p_load_mw: pd.DataFrame
    q_load_mvar: pd.DataFrame
    source: str
    metadata: dict[str, object]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 24-hour Swissgrid 2025 contingency-risk assessment and "
            "produce the 6x4 voltage/loading network figure."
        )
    )
    parser.add_argument(
        "--system-dir",
        type=Path,
        default=ROOT / "data" / "powersystems" / "Swiss2025",
        help="Swiss2025 data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "swisscase2025" / "risk_24h",
        help="Output directory.",
    )
    parser.add_argument(
        "--profile-source",
        choices=("auto", "hourly-demand", "eua"),
        default="auto",
        help=(
            "Operating-profile source. 'auto' first uses hourlyDemandBus and "
            "then falls back to EuA P/Q text files."
        ),
    )
    parser.add_argument(
        "--p-file",
        type=Path,
        default=None,
        help="Active-power EuA text file, e.g. 20250425_P_EuA_*.txt.",
    )
    parser.add_argument(
        "--q-file",
        type=Path,
        default=None,
        help="Reactive-power EuA text file, e.g. 20250425_Q_EuA_*.txt.",
    )
    parser.add_argument(
        "--eua-mapping-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit mapping with columns eua_id and load_index. "
            "Optional columns: p_sign and q_sign."
        ),
    )
    parser.add_argument(
        "--minimum-mapped-load-fraction",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of nominal active load that must be mapped from "
            "the EuA files. The script fails below this threshold."
        ),
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="First row of the demand time series to use.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of consecutive operating points. The reference figure uses 24.",
    )
    parser.add_argument(
        "--load-scale",
        type=float,
        default=1.0,
        help="Multiplicative scaling applied to all active-power demand profiles.",
    )
    parser.add_argument(
        "--base-pf-mode",
        choices=("ac", "dc"),
        default="ac",
        help="Power-flow model used for the figure's base-case states.",
    )
    parser.add_argument(
        "--risk-pf-mode",
        choices=("ac", "dc"),
        default="dc",
        help="Power-flow model used for the N-1 contingency analysis.",
    )
    parser.add_argument(
        "--allow-dc-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DC power flow when an AC base-case power flow does not converge.",
    )
    parser.add_argument(
        "--contingency-types",
        nargs="+",
        choices=("line", "trafo", "gen"),
        default=("line", "gen"),
        help="Element types included in the N-1 contingency catalogue.",
    )
    parser.add_argument(
        "--max-contingencies",
        type=int,
        default=None,
        help=(
            "Optional deterministic truncation of the contingency list. "
            "Use only for debugging; omit for the complete selected N-1 set."
        ),
    )
    parser.add_argument(
        "--total-contingency-probability",
        type=float,
        default=0.01,
        help=(
            "Total probability mass assigned uniformly to all selected N-1 "
            "contingencies. The remaining mass is assigned to the base case."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Confidence level used for VaR and CVaR.",
    )
    parser.add_argument(
        "--loading-threshold",
        type=float,
        default=100.0,
        help="Branch loading threshold used to define overload severity.",
    )
    parser.add_argument(
        "--severity-metric",
        choices=("overload_excess", "n_overloads", "voltage_violation", "combined"),
        default="overload_excess",
        help="Scalar consequence metric used for PRA aggregation.",
    )
    parser.add_argument(
        "--map-mode",
        choices=("base", "expected", "worst"),
        default="base",
        help=(
            "Values displayed in the 24-panel figure: base-case values, "
            "probability-weighted post-contingency expectations, or worst values."
        ),
    )
    parser.add_argument(
        "--line-loading-vmax",
        type=float,
        default=20.0,
        help="Upper line-loading colour limit. The supplied figure uses 20%%.",
    )
    parser.add_argument(
        "--voltage-vmin",
        type=float,
        default=0.95,
        help="Lower bus-voltage colour limit.",
    )
    parser.add_argument(
        "--voltage-vmax",
        type=float,
        default=1.05,
        help="Upper bus-voltage colour limit.",
    )
    parser.add_argument(
        "--show-hour-titles",
        action="store_true",
        help="Add an hour label above every panel. Disabled to match the example.",
    )
    parser.add_argument(
        "--skip-risk",
        action="store_true",
        help="Produce the base-state figure without running N-1 analysis.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display contingency-analysis progress.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def validate_args(args: argparse.Namespace) -> None:
    if args.hours <= 0:
        raise ValueError("--hours must be positive.")
    if args.start_hour < 0:
        raise ValueError("--start-hour cannot be negative.")
    if args.load_scale <= 0.0:
        raise ValueError("--load-scale must be positive.")
    if not 0.0 <= args.total_contingency_probability <= 1.0:
        raise ValueError("--total-contingency-probability must lie in [0, 1].")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie strictly between zero and one.")
    if args.line_loading_vmax <= 0.0:
        raise ValueError("--line-loading-vmax must be positive.")
    if args.voltage_vmin >= args.voltage_vmax:
        raise ValueError("--voltage-vmin must be smaller than --voltage-vmax.")
    if not 0.0 <= args.minimum_mapped_load_fraction <= 1.0:
        raise ValueError("--minimum-mapped-load-fraction must lie in [0, 1].")



def _normalise_identifier(value: object) -> str:
    """Return a compact identifier for robust exact matching."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def read_eua_profile_file(path: Path) -> tuple[str | None, pd.DataFrame]:
    """Parse a semicolon-separated EuA P or Q profile file.

    The first line starts with ``!`` and contains metadata. Every subsequent
    row contains an equipment identifier followed by 24 hourly values.
    """

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"EuA profile file not found: {path}")

    header: str | None = None
    identifiers: list[str] = []
    values: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("!"):
                header = line[1:].rstrip(";")
                continue

            parts = line.split(";")
            if parts and parts[-1] == "":
                parts = parts[:-1]
            if len(parts) < 2:
                raise ValueError(f"Malformed EuA row at {path}:{line_number}")

            identifier = parts[0].strip()
            row_values = parts[1:]
            try:
                numeric = [float(item.replace(",", ".")) for item in row_values]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric EuA value at {path}:{line_number}"
                ) from exc

            identifiers.append(identifier)
            values.append(numeric)

    if not values:
        raise ValueError(f"No EuA profile rows were found in {path}.")

    n_steps = len(values[0])
    inconsistent = [
        (identifier, len(row))
        for identifier, row in zip(identifiers, values)
        if len(row) != n_steps
    ]
    if inconsistent:
        raise ValueError(
            f"Inconsistent EuA row lengths in {path}; examples: {inconsistent[:5]}"
        )
    if len(set(identifiers)) != len(identifiers):
        duplicates = pd.Series(identifiers).value_counts()
        duplicate_ids = duplicates.loc[duplicates > 1].index.tolist()
        raise ValueError(
            f"Duplicate EuA identifiers in {path}: {duplicate_ids[:10]}"
        )

    frame = pd.DataFrame(
        values,
        index=pd.Index(identifiers, name="eua_id"),
        columns=pd.RangeIndex(n_steps, name="hour"),
        dtype=float,
    )
    return header, frame


def _find_eua_file(
    *,
    explicit_path: Path | None,
    system_dir: Path,
    pattern: str,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    candidates = sorted(system_dir.glob(pattern))
    if not candidates:
        candidates = sorted(ROOT.glob(pattern))
    if not candidates:
        return None
    if len(candidates) > 1:
        LOGGER.warning(
            "Several files match %s; using %s",
            pattern,
            candidates[-1],
        )
    return candidates[-1].resolve()


def nominal_q_over_p(net) -> pd.Series:
    p = pd.to_numeric(net.load["p_mw"], errors="coerce").fillna(0.0)
    q = pd.to_numeric(net.load.get("q_mvar", 0.0), errors="coerce")
    if not isinstance(q, pd.Series):
        q = pd.Series(float(q), index=net.load.index)
    q = q.reindex(net.load.index).fillna(0.0)

    ratio = pd.Series(0.0, index=net.load.index, dtype=float)
    mask = p.abs() > 1e-12
    ratio.loc[mask] = q.loc[mask] / p.loc[mask]
    return ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _load_string_candidates(net, load_index: int) -> set[str]:
    """Collect exact source identifiers stored on a pandapower load row."""

    candidates: set[str] = set()
    row = net.load.loc[load_index]

    for column, value in row.items():
        if pd.api.types.is_object_dtype(net.load[column].dtype) or isinstance(value, str):
            token = _normalise_identifier(value)
            if token:
                candidates.add(token)

    # A bus-level source identifier is safe only when a single load is connected
    # to the bus; otherwise it would create an ambiguous many-to-one mapping.
    bus_index = int(row["bus"])
    loads_at_bus = net.load.index[net.load["bus"] == bus_index]
    if len(loads_at_bus) == 1 and bus_index in net.bus.index:
        bus_row = net.bus.loc[bus_index]
        for column, value in bus_row.items():
            if pd.api.types.is_object_dtype(net.bus[column].dtype) or isinstance(value, str):
                token = _normalise_identifier(value)
                if token:
                    candidates.add(token)

    return candidates


def _read_explicit_eua_mapping(path: Path, net) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    required = {"eua_id", "load_index"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(
            f"Mapping file {path} is missing required columns: {sorted(missing)}"
        )

    mapping = mapping.copy()
    mapping["eua_id"] = mapping["eua_id"].astype(str)
    mapping["load_index"] = mapping["load_index"].astype(int)

    unknown = sorted(set(mapping["load_index"]) - set(map(int, net.load.index)))
    if unknown:
        raise ValueError(f"Mapping file contains unknown load indices: {unknown[:10]}")

    if mapping["load_index"].duplicated().any():
        duplicate_loads = (
            mapping.loc[mapping["load_index"].duplicated(False), "load_index"]
            .astype(int)
            .tolist()
        )
        raise ValueError(
            "Each pandapower load must map to at most one EuA identifier. "
            f"Duplicate load indices: {duplicate_loads[:10]}"
        )

    if "p_sign" not in mapping:
        mapping["p_sign"] = np.nan
    if "q_sign" not in mapping:
        mapping["q_sign"] = np.nan
    return mapping


def build_eua_load_mapping(
    net,
    p_table: pd.DataFrame,
    *,
    mapping_csv: Path | None,
) -> pd.DataFrame:
    """Map EuA identifiers to pandapower loads.

    Explicit mappings take priority. Otherwise the function performs strict
    normalised exact matching against string-valued load metadata and, where
    unambiguous, the corresponding bus metadata.
    """

    if mapping_csv is not None:
        mapping = _read_explicit_eua_mapping(mapping_csv.resolve(), net)
        source = "explicit"
    else:
        normalised_to_ids: dict[str, list[str]] = {}
        for identifier in p_table.index:
            normalised_to_ids.setdefault(
                _normalise_identifier(identifier), []
            ).append(str(identifier))

        records: list[dict[str, object]] = []
        for load_index in net.load.index:
            matches: set[str] = set()
            for token in _load_string_candidates(net, int(load_index)):
                candidates = normalised_to_ids.get(token, [])
                if len(candidates) == 1:
                    matches.add(candidates[0])

            if len(matches) == 1:
                records.append(
                    {
                        "eua_id": matches.pop(),
                        "load_index": int(load_index),
                        "p_sign": np.nan,
                        "q_sign": np.nan,
                    }
                )
            elif len(matches) > 1:
                LOGGER.warning(
                    "Ambiguous EuA identifiers for load %s: %s",
                    load_index,
                    sorted(matches),
                )

        mapping = pd.DataFrame(
            records,
            columns=["eua_id", "load_index", "p_sign", "q_sign"],
        )
        source = "automatic-exact"

    if mapping.empty:
        mapping.attrs["source"] = source
        return mapping

    missing_ids = sorted(set(mapping["eua_id"]) - set(p_table.index))
    if missing_ids:
        raise ValueError(
            f"Mapped EuA identifiers are absent from the P file: {missing_ids[:10]}"
        )

    mapping.attrs["source"] = source
    return mapping


def _infer_sign(values: pd.Series, nominal_value: float, fallback: float = 1.0) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    finite = finite.loc[finite.abs() > 1e-9]
    if finite.empty:
        return float(fallback)

    median_value = float(finite.median())
    if abs(nominal_value) > 1e-9:
        return 1.0 if median_value * nominal_value >= 0.0 else -1.0
    return 1.0 if median_value >= 0.0 else -1.0


def _convert_bus_or_load_demand_to_load_profiles(
    demand: pd.DataFrame,
    net,
) -> pd.DataFrame:
    """Convert an existing hourlyDemandBus table to one column per net.load."""

    demand = demand.apply(pd.to_numeric, errors="coerce")
    if demand.isna().any().any():
        bad = int(demand.isna().sum().sum())
        raise ValueError(f"The demand table contains {bad} invalid values.")

    if demand.shape[1] == len(net.load):
        out = demand.copy()
        out.columns = net.load.index
        return out

    if demand.shape[1] != len(net.bus):
        raise ValueError(
            f"Demand table has {demand.shape[1]} columns, but the network has "
            f"{len(net.load)} loads and {len(net.bus)} buses."
        )

    nominal_p = pd.to_numeric(net.load["p_mw"], errors="coerce").fillna(0.0)
    out = pd.DataFrame(0.0, index=demand.index, columns=net.load.index)

    for position, bus_index in enumerate(net.bus.index):
        load_indices = net.load.index[net.load["bus"] == bus_index]
        if len(load_indices) == 0:
            continue

        base = nominal_p.loc[load_indices].abs().to_numpy(dtype=float)
        denominator = float(base.sum())
        weights = (
            base / denominator
            if denominator > 1e-12
            else np.full(len(load_indices), 1.0 / len(load_indices))
        )
        out.loc[:, load_indices] = (
            demand.iloc[:, position].to_numpy(dtype=float)[:, None]
            * weights[None, :]
        )
    return out


def prepare_profiles_from_hourly_demand(
    demand: pd.DataFrame,
    net,
    *,
    start_hour: int,
    hours: int,
    load_scale: float,
) -> PreparedProfiles:
    end = start_hour + hours
    if end > len(demand):
        raise ValueError(
            f"Requested rows [{start_hour}:{end}], but the demand table has "
            f"only {len(demand)} rows."
        )

    selected = demand.iloc[start_hour:end].copy()
    p_load = _convert_bus_or_load_demand_to_load_profiles(selected, net)
    p_load *= float(load_scale)
    p_load.index = pd.RangeIndex(hours, name="hour")

    q_ratio = nominal_q_over_p(net).reindex(p_load.columns).fillna(0.0)
    q_load = p_load.mul(q_ratio, axis=1)

    return PreparedProfiles(
        p_load_mw=p_load,
        q_load_mvar=q_load,
        source="hourly-demand",
        metadata={"rows_selected": [start_hour, end]},
    )


def prepare_profiles_from_eua(
    net,
    *,
    p_file: Path,
    q_file: Path | None,
    mapping_csv: Path | None,
    start_hour: int,
    hours: int,
    load_scale: float,
    minimum_mapped_load_fraction: float,
    output_dir: Path,
) -> PreparedProfiles:
    p_header, p_table = read_eua_profile_file(p_file)
    q_header: str | None = None
    q_table: pd.DataFrame | None = None
    if q_file is not None:
        q_header, q_table = read_eua_profile_file(q_file)

    end = start_hour + hours
    if end > p_table.shape[1]:
        raise ValueError(
            f"Requested EuA hours [{start_hour}:{end}], but the P file contains "
            f"only {p_table.shape[1]} values per identifier."
        )
    if q_table is not None and end > q_table.shape[1]:
        raise ValueError(
            f"Requested EuA hours [{start_hour}:{end}], but the Q file contains "
            f"only {q_table.shape[1]} values per identifier."
        )

    mapping = build_eua_load_mapping(
        net,
        p_table,
        mapping_csv=mapping_csv,
    )

    nominal_p = pd.to_numeric(net.load["p_mw"], errors="coerce").fillna(0.0)
    nominal_q = pd.to_numeric(net.load.get("q_mvar", 0.0), errors="coerce")
    if not isinstance(nominal_q, pd.Series):
        nominal_q = pd.Series(float(nominal_q), index=net.load.index)
    nominal_q = nominal_q.reindex(net.load.index).fillna(0.0)

    nominal_total = float(nominal_p.abs().sum())
    mapped_indices = (
        mapping["load_index"].astype(int).tolist()
        if not mapping.empty
        else []
    )
    mapped_nominal = float(nominal_p.loc[mapped_indices].abs().sum())
    mapped_fraction = (
        mapped_nominal / nominal_total
        if nominal_total > 1e-12
        else len(mapped_indices) / max(len(net.load), 1)
    )

    report_rows: list[dict[str, object]] = []
    mapped_by_index = (
        mapping.set_index("load_index").to_dict("index")
        if not mapping.empty
        else {}
    )
    for load_index in net.load.index:
        record = mapped_by_index.get(int(load_index))
        report_rows.append(
            {
                "load_index": int(load_index),
                "load_name": net.load.at[load_index, "name"]
                if "name" in net.load.columns
                else None,
                "bus": int(net.load.at[load_index, "bus"]),
                "nominal_p_mw": float(nominal_p.at[load_index]),
                "eua_id": None if record is None else record["eua_id"],
                "mapping_status": "mapped" if record is not None else "unmapped",
            }
        )
    mapping_report = pd.DataFrame(report_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_report.to_csv(output_dir / "eua_load_mapping_report.csv", index=False)

    if mapping.empty or mapped_fraction < minimum_mapped_load_fraction:
        raise RuntimeError(
            "The EuA P/Q files were parsed successfully, but they are equipment-"
            "level profiles and could not be mapped reliably to enough pandapower "
            "loads.\n"
            f"Mapped loads: {len(mapped_indices)}/{len(net.load)}; "
            f"nominal-MW coverage: {mapped_fraction:.1%}; required: "
            f"{minimum_mapped_load_fraction:.1%}.\n"
            f"Inspect {output_dir / 'eua_load_mapping_report.csv'} and provide "
            "--eua-mapping-csv with columns eua_id,load_index,p_sign,q_sign."
        )

    p_load = pd.DataFrame(
        np.tile(nominal_p.to_numpy(dtype=float), (hours, 1)),
        index=pd.RangeIndex(hours, name="hour"),
        columns=net.load.index,
    )
    q_load = pd.DataFrame(
        np.tile(nominal_q.to_numpy(dtype=float), (hours, 1)),
        index=pd.RangeIndex(hours, name="hour"),
        columns=net.load.index,
    )

    for row in mapping.itertuples(index=False):
        load_index = int(row.load_index)
        eua_id = str(row.eua_id)

        raw_p = p_table.loc[eua_id].iloc[start_hour:end].reset_index(drop=True)
        p_sign = (
            float(row.p_sign)
            if pd.notna(row.p_sign)
            else _infer_sign(raw_p, float(nominal_p.at[load_index]))
        )
        converted_p = p_sign * raw_p.to_numpy(dtype=float) * float(load_scale)
        p_load.loc[:, load_index] = converted_p

        if q_table is not None and eua_id in q_table.index:
            raw_q = q_table.loc[eua_id].iloc[start_hour:end].reset_index(drop=True)
            q_sign = (
                float(row.q_sign)
                if pd.notna(row.q_sign)
                else _infer_sign(
                    raw_q,
                    float(nominal_q.at[load_index]),
                    fallback=p_sign,
                )
            )
            q_load.loc[:, load_index] = (
                q_sign * raw_q.to_numpy(dtype=float) * float(load_scale)
            )
        else:
            ratio = (
                float(nominal_q.at[load_index] / nominal_p.at[load_index])
                if abs(float(nominal_p.at[load_index])) > 1e-12
                else 0.0
            )
            q_load.loc[:, load_index] = converted_p * ratio

    if (p_load.to_numpy(dtype=float) < -1e-6).any():
        LOGGER.warning(
            "Some mapped load profiles are negative. They are retained because "
            "pandapower supports negative loads as net injections. Verify the "
            "p_sign values in the mapping report."
        )

    metadata = {
        "p_file": str(p_file),
        "q_file": None if q_file is None else str(q_file),
        "p_header": p_header,
        "q_header": q_header,
        "p_identifiers": int(len(p_table)),
        "q_identifiers": 0 if q_table is None else int(len(q_table)),
        "mapping_source": mapping.attrs.get("source", "unknown"),
        "mapped_load_count": int(len(mapped_indices)),
        "total_load_count": int(len(net.load)),
        "mapped_nominal_load_fraction": float(mapped_fraction),
    }

    LOGGER.info(
        "Prepared EuA profiles: %d P identifiers, %d Q identifiers, "
        "%d/%d loads mapped (%.1f%% nominal-MW coverage).",
        len(p_table),
        0 if q_table is None else len(q_table),
        len(mapped_indices),
        len(net.load),
        100.0 * mapped_fraction,
    )

    return PreparedProfiles(
        p_load_mw=p_load,
        q_load_mvar=q_load,
        source="eua",
        metadata=metadata,
    )


def load_profiles(
    system_dir: Path,
    net,
    *,
    profile_source: str,
    p_file: Path | None,
    q_file: Path | None,
    mapping_csv: Path | None,
    start_hour: int,
    hours: int,
    load_scale: float,
    minimum_mapped_load_fraction: float,
    output_dir: Path,
) -> PreparedProfiles:
    """Load hourlyDemandBus or EuA P/Q profiles and align them with net.load."""

    demand = None
    if profile_source in {"auto", "hourly-demand"}:
        demand = load_hourly_demand(system_dir)
        if demand is not None and not demand.empty:
            LOGGER.info("Using hourly demand data from %s.", system_dir)
            return prepare_profiles_from_hourly_demand(
                demand,
                net,
                start_hour=start_hour,
                hours=hours,
                load_scale=load_scale,
            )
        if profile_source == "hourly-demand":
            raise FileNotFoundError(
                "No hourly demand data were found. Expected one of:\n"
                f"  {system_dir / 'hourlyDemandBus.pkl'}\n"
                f"  {system_dir / 'hourlyDemandBus.mat'}\n"
                f"  {system_dir / 'System.xlsx'} with sheet 'hourlyDemandBus'."
            )

    resolved_p = _find_eua_file(
        explicit_path=p_file,
        system_dir=system_dir,
        pattern="*_P_EuA_*.txt",
    )
    resolved_q = _find_eua_file(
        explicit_path=q_file,
        system_dir=system_dir,
        pattern="*_Q_EuA_*.txt",
    )

    if resolved_p is None:
        raise FileNotFoundError(
            "No hourlyDemandBus data or EuA active-power file was found.\n"
            "Pass --p-file and optionally --q-file, or copy the EuA files into:\n"
            f"  {system_dir}"
        )

    LOGGER.info("Using EuA active-power file: %s", resolved_p)
    if resolved_q is None:
        LOGGER.warning(
            "No EuA Q file was found; reactive demand will use nominal q/p ratios."
        )
    else:
        LOGGER.info("Using EuA reactive-power file: %s", resolved_q)

    return prepare_profiles_from_eua(
        net,
        p_file=resolved_p,
        q_file=resolved_q,
        mapping_csv=mapping_csv,
        start_hour=start_hour,
        hours=hours,
        load_scale=load_scale,
        minimum_mapped_load_fraction=minimum_mapped_load_fraction,
        output_dir=output_dir,
    )


def apply_load_profile(
    net,
    p_profile: pd.Series,
    q_profile: pd.Series | None,
) -> None:
    """Apply one load-level P/Q state aligned with net.load indices."""

    p = pd.to_numeric(p_profile.reindex(net.load.index), errors="coerce")
    if p.isna().any():
        missing = p.index[p.isna()].tolist()
        raise ValueError(f"Active-load profile is missing load indices: {missing[:10]}")
    net.load.loc[:, "p_mw"] = p.to_numpy(dtype=float)

    if "q_mvar" in net.load.columns and q_profile is not None:
        q = pd.to_numeric(q_profile.reindex(net.load.index), errors="coerce")
        if q.isna().any():
            missing = q.index[q.isna()].tolist()
            raise ValueError(f"Reactive-load profile is missing load indices: {missing[:10]}")
        net.load.loc[:, "q_mvar"] = q.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Base-case power flows used for the reference figure
# ---------------------------------------------------------------------------
def run_base_power_flow(
    net,
    p_profile: pd.Series,
    q_profile: pd.Series,
    *,
    mode: str,
    allow_dc_fallback: bool,
) -> tuple[object, bool, str]:
    """Run one base-case power flow on a clean network copy."""

    case = copy.deepcopy(net)
    apply_load_profile(case, p_profile, q_profile)

    mode = mode.lower()
    try:
        if mode == "ac":
            pp.runpp(
                case,
                calculate_voltage_angles=True,
                init="auto",
                enforce_q_lims=True,
            )
        else:
            pp.rundcpp(case)
        return case, bool(getattr(case, "converged", True)), mode
    except Exception as exc:
        if mode == "ac" and allow_dc_fallback:
            LOGGER.warning("AC power flow failed (%s); trying DC fallback.", exc)
            case = copy.deepcopy(net)
            apply_load_profile(case, p_profile, q_profile)
            pp.rundcpp(case)
            return case, bool(getattr(case, "converged", True)), "dc-fallback"
        raise RuntimeError(f"{mode.upper()} base-case power flow failed: {exc}") from exc


def collect_base_states(
    net,
    profiles: PreparedProfiles,
    *,
    mode: str,
    allow_dc_fallback: bool,
) -> tuple[list[HourlyMapState], pd.DataFrame]:
    states: list[HourlyMapState] = []
    rows: list[dict[str, float | int | str | bool]] = []

    for hour in profiles.p_load_mw.index:
        p_profile = profiles.p_load_mw.loc[hour]
        q_profile = profiles.q_load_mvar.loc[hour]
        solved, converged, mode_used = run_base_power_flow(
            net,
            p_profile,
            q_profile,
            mode=mode,
            allow_dc_fallback=allow_dc_fallback,
        )

        line_loading = pd.to_numeric(
            solved.res_line.get("loading_percent", pd.Series(index=solved.line.index, dtype=float)),
            errors="coerce",
        ).reindex(solved.line.index)

        bus_voltage = pd.to_numeric(
            solved.res_bus.get("vm_pu", pd.Series(index=solved.bus.index, dtype=float)),
            errors="coerce",
        ).reindex(solved.bus.index)

        states.append(
            HourlyMapState(
                hour=hour,
                line_loading_pct=line_loading,
                bus_voltage_pu=bus_voltage,
                converged=converged,
                pf_mode_used=mode_used,
            )
        )
        rows.append(
            {
                "hour": hour,
                "pf_mode_used": mode_used,
                "converged": converged,
                "total_demand_mw": float(p_profile.sum()),
                "total_reactive_demand_mvar": float(q_profile.sum()),
                "max_line_loading_pct": float(line_loading.max(skipna=True)),
                "mean_line_loading_pct": float(line_loading.mean(skipna=True)),
                "min_bus_voltage_pu": float(bus_voltage.min(skipna=True)),
                "max_bus_voltage_pu": float(bus_voltage.max(skipna=True)),
            }
        )

    return states, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# N-1 probabilistic risk assessment
# ---------------------------------------------------------------------------
def build_contingency_set(
    net,
    *,
    element_types: Sequence[str],
    max_contingencies: int | None,
):
    contingencies = build_n1_contingencies(
        net,
        include=tuple(element_types),
        only_in_service=True,
    )
    contingencies = sorted(contingencies, key=lambda c: str(c.contingency_id))

    if max_contingencies is not None:
        if max_contingencies <= 0:
            raise ValueError("--max-contingencies must be positive.")
        contingencies = contingencies[:max_contingencies]

    if not contingencies:
        raise ValueError("The selected contingency catalogue is empty.")
    return contingencies


def run_risk_assessment(
    net,
    profiles: PreparedProfiles,
    *,
    contingency_types: Sequence[str],
    max_contingencies: int | None,
    total_contingency_probability: float,
    alpha: float,
    loading_threshold: float,
    severity_metric: str,
    pf_mode: str,
    progress: bool,
    output_dir: Path,
):
    contingencies = build_contingency_set(
        net,
        element_types=contingency_types,
        max_contingencies=max_contingencies,
    )
    LOGGER.info(
        "Running %d N-1 contingencies over %d operating points.",
        len(contingencies),
        len(profiles.p_load_mw),
    )

    analysis = run_contingency_analysis(
        net,
        operating_points=profiles.p_load_mw,
        contingencies=contingencies,
        pf_mode=pf_mode,
        include_base_case=True,
        loading_threshold=loading_threshold,
        severity_metric=severity_metric,
        progress=progress,
    )

    probabilities = UniformContingencyModel(
        total_contingency_probability=total_contingency_probability
    ).get_probabilities(contingencies)

    risk = aggregate_risk(
        analysis.summary,
        probabilities=probabilities,
        severity_col="severity",
        time_col="operating_point_id",
        contingency_col="contingency_id",
        base_case_id="base_case",
        alpha=alpha,
    )

    analysis_dir = output_dir / "contingency_analysis"
    risk_dir = output_dir / "risk"
    analysis.to_csv(analysis_dir)
    risk.to_csv(risk_dir)

    probability_table = pd.DataFrame(
        [
            {"contingency_id": "base_case", "probability": 1.0 - sum(probabilities.values())},
            *[
                {"contingency_id": contingency_id, "probability": probability}
                for contingency_id, probability in sorted(probabilities.items())
            ],
        ]
    )
    probability_table.to_csv(output_dir / "contingency_probabilities.csv", index=False)

    return analysis, risk, probabilities


# ---------------------------------------------------------------------------
# Convert contingency outputs to map values
# ---------------------------------------------------------------------------
def case_probability_mapping(
    probabilities: dict[str, float],
) -> dict[str, float]:
    total = float(sum(probabilities.values()))
    return {"base_case": max(0.0, 1.0 - total), **probabilities}


def weighted_finite_mean(values: pd.Series, weights: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    if not mask.any() or float(w[mask].sum()) <= 0.0:
        return float("nan")
    return float(np.dot(x[mask], w[mask]) / w[mask].sum())


def most_extreme_voltage(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(x[np.argmax(np.abs(x - 1.0))])


def contingency_map_states(
    net,
    analysis,
    probabilities: dict[str, float],
    *,
    mode: str,
    hours: int,
) -> list[HourlyMapState]:
    """Build expected or worst post-contingency map states."""

    if mode not in {"expected", "worst"}:
        raise ValueError("Contingency map mode must be 'expected' or 'worst'.")

    pmap = case_probability_mapping(probabilities)

    components = analysis.components.copy()
    buses = analysis.bus.copy()
    if components.empty:
        raise RuntimeError("No contingency component results are available for mapping.")
    if buses.empty:
        raise RuntimeError("No contingency bus results are available for mapping.")

    components["case_probability"] = components["contingency_id"].map(pmap).fillna(0.0)
    buses["case_probability"] = buses["contingency_id"].map(pmap).fillna(0.0)

    line_rows = components.loc[components["element_type"] == "line"].copy()

    states: list[HourlyMapState] = []
    for hour in range(hours):
        hourly_lines = line_rows.loc[line_rows["operating_point_id"] == hour]
        hourly_buses = buses.loc[buses["operating_point_id"] == hour]

        if mode == "expected":
            line_values = (
                hourly_lines.groupby("element_index", sort=False)
                .apply(
                    lambda group: weighted_finite_mean(
                        group["loading_percent"], group["case_probability"]
                    )
                )
                .reindex(net.line.index)
            )
            bus_values = (
                hourly_buses.groupby("bus", sort=False)
                .apply(
                    lambda group: weighted_finite_mean(
                        group["vm_pu"], group["case_probability"]
                    )
                )
                .reindex(net.bus.index)
            )
        else:
            line_values = (
                hourly_lines.groupby("element_index")["loading_percent"]
                .max()
                .reindex(net.line.index)
            )
            bus_values = (
                hourly_buses.groupby("bus")["vm_pu"]
                .apply(most_extreme_voltage)
                .reindex(net.bus.index)
            )

        states.append(
            HourlyMapState(
                hour=hour,
                line_loading_pct=pd.to_numeric(line_values, errors="coerce"),
                bus_voltage_pu=pd.to_numeric(bus_values, errors="coerce"),
                converged=True,
                pf_mode_used=f"contingency-{mode}",
            )
        )

    return states


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def bus_coordinates(net) -> pd.DataFrame:
    if not hasattr(net, "bus_geodata") or net.bus_geodata is None or net.bus_geodata.empty:
        raise ValueError(
            "The Swiss2025 network has no bus_geodata. "
            "The local case builder must attach longitude/latitude coordinates."
        )

    xy = net.bus_geodata[["x", "y"]].copy()
    xy["x"] = pd.to_numeric(xy["x"], errors="coerce")
    xy["y"] = pd.to_numeric(xy["y"], errors="coerce")
    return xy.dropna(subset=["x", "y"])


def line_segments(net, xy: pd.DataFrame) -> tuple[list[list[tuple[float, float]]], list[int]]:
    segments: list[list[tuple[float, float]]] = []
    indices: list[int] = []

    for line_idx, row in net.line.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        if from_bus not in xy.index or to_bus not in xy.index:
            continue
        segments.append(
            [
                (float(xy.at[from_bus, "x"]), float(xy.at[from_bus, "y"])),
                (float(xy.at[to_bus, "x"]), float(xy.at[to_bus, "y"])),
            ]
        )
        indices.append(int(line_idx))
    return segments, indices


def add_compact_colorbar(
    fig,
    ax,
    *,
    mappable,
    bounds: tuple[float, float, float, float],
    label: str,
) -> None:
    cax = ax.inset_axes(bounds)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.6)
    cbar.set_label(label, fontsize=7, labelpad=2)


def plot_24_hour_states(
    net,
    states: Sequence[HourlyMapState],
    *,
    output_png: Path,
    output_pdf: Path,
    line_loading_vmax: float,
    voltage_vmin: float,
    voltage_vmax: float,
    show_hour_titles: bool,
) -> None:
    """Create the 4-column network panel layout shown in the reference image."""

    xy = bus_coordinates(net)
    segments, valid_line_indices = line_segments(net, xy)
    if not segments:
        raise ValueError("No valid line segments could be constructed.")

    ncols = 4
    nrows = int(math.ceil(len(states) / ncols))

    # The 24-hour reference image is approximately 17.3 x 15.6 inches at 100 dpi.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(17.3, 2.60 * nrows),
        squeeze=False,
    )

    voltage_norm = Normalize(vmin=voltage_vmin, vmax=voltage_vmax, clip=True)
    loading_norm = Normalize(vmin=0.0, vmax=line_loading_vmax, clip=True)
    voltage_cmap = plt.get_cmap("RdYlGn")
    loading_cmap = plt.get_cmap("RdYlGn_r")

    x_min = float(xy["x"].min())
    x_max = float(xy["x"].max())
    y_min = float(xy["y"].min())
    y_max = float(xy["y"].max())
    x_margin = 0.03 * max(x_max - x_min, 1e-6)
    y_margin = 0.03 * max(y_max - y_min, 1e-6)

    for panel, ax in enumerate(axes.flat):
        if panel >= len(states):
            ax.axis("off")
            continue

        state = states[panel]

        line_values = (
            state.line_loading_pct.reindex(valid_line_indices)
            .to_numpy(dtype=float)
        )
        line_rgba = loading_cmap(
            loading_norm(np.nan_to_num(line_values, nan=0.0))
        )
        line_rgba[~np.isfinite(line_values), 3] = 0.0

        # Grey network underlay, followed by loading-coloured lines.
        ax.add_collection(
            LineCollection(
                segments,
                colors="0.62",
                linewidths=1.05,
                alpha=0.80,
                zorder=1,
            )
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=line_rgba,
                linewidths=1.25,
                alpha=0.88,
                zorder=2,
            )
        )

        bus_values = state.bus_voltage_pu.reindex(xy.index).to_numpy(dtype=float)
        scatter = ax.scatter(
            xy["x"],
            xy["y"],
            c=bus_values,
            cmap=voltage_cmap,
            norm=voltage_norm,
            s=16,
            edgecolors="none",
            zorder=3,
        )

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if show_hour_titles:
            ax.set_title(f"Hour {state.hour:02d}", fontsize=8, pad=1)

        voltage_sm = ScalarMappable(norm=voltage_norm, cmap=voltage_cmap)
        voltage_sm.set_array(bus_values)
        loading_sm = ScalarMappable(norm=loading_norm, cmap=loading_cmap)
        loading_sm.set_array(line_values)

        # Positions reproduce the two narrow vertical bars in each panel.
        add_compact_colorbar(
            fig,
            ax,
            mappable=voltage_sm,
            bounds=(0.70, 0.10, 0.026, 0.80),
            label="Bus Voltage [pu]",
        )
        add_compact_colorbar(
            fig,
            ax,
            mappable=loading_sm,
            bounds=(0.87, 0.10, 0.026, 0.80),
            label="Line Loading [%]",
        )

    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        bottom=0.01,
        top=0.995,
        wspace=0.09,
        hspace=0.10,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=100, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_run_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    net,
    profiles: PreparedProfiles,
    contingency_count: int | None,
) -> None:
    metadata = {
        "system": "Swissgrid 2025 research case",
        "n_buses": int(len(net.bus)),
        "n_lines": int(len(net.line)),
        "n_transformers": int(len(getattr(net, "trafo", []))),
        "n_generators": int(len(getattr(net, "gen", []))),
        "n_static_generators": int(len(getattr(net, "sgen", []))),
        "n_loads": int(len(net.load)),
        "operating_points": int(len(profiles.p_load_mw)),
        "profile_source": profiles.source,
        "profile_metadata": profiles.metadata,
        "start_hour": int(args.start_hour),
        "load_scale": float(args.load_scale),
        "base_pf_mode": args.base_pf_mode,
        "risk_pf_mode": args.risk_pf_mode,
        "contingency_types": list(args.contingency_types),
        "contingency_count": contingency_count,
        "total_contingency_probability": float(args.total_contingency_probability),
        "risk_alpha": float(args.alpha),
        "severity_metric": args.severity_metric,
        "loading_threshold_pct": float(args.loading_threshold),
        "map_mode": args.map_mode,
        "planned_outages": [],
        "risk_scope": (
            "Intact-system base case and N-1 contingencies; "
            "no planned maintenance outage is applied."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args.system_dir = args.system_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_path = args.system_dir / "case_swissgrid2025.py"
    if not case_path.exists():
        raise FileNotFoundError(
            f"Missing NDA-protected Swissgrid case builder: {case_path}"
        )

    LOGGER.info("Loading Swissgrid 2025 research case.")
    net = case_swissgrid2025()
    profiles = load_profiles(
        args.system_dir,
        net,
        profile_source=args.profile_source,
        p_file=args.p_file,
        q_file=args.q_file,
        mapping_csv=args.eua_mapping_csv,
        start_hour=args.start_hour,
        hours=args.hours,
        load_scale=args.load_scale,
        minimum_mapped_load_fraction=args.minimum_mapped_load_fraction,
        output_dir=args.output_dir,
    )

    LOGGER.info(
        "System: %d buses, %d lines, %d generators, %d loads.",
        len(net.bus),
        len(net.line),
        len(getattr(net, "gen", [])),
        len(net.load),
    )

    LOGGER.info(
        "Solving %d base operating states from %s profiles.",
        len(profiles.p_load_mw),
        profiles.source,
    )
    base_states, base_metrics = collect_base_states(
        net,
        profiles,
        mode=args.base_pf_mode,
        allow_dc_fallback=args.allow_dc_fallback,
    )
    base_metrics.to_csv(args.output_dir / "base_hourly_metrics.csv", index=False)

    analysis = None
    risk = None
    probabilities: dict[str, float] = {}
    contingency_count: int | None = None

    if not args.skip_risk:
        analysis, risk, probabilities = run_risk_assessment(
            net,
            profiles,
            contingency_types=args.contingency_types,
            max_contingencies=args.max_contingencies,
            total_contingency_probability=args.total_contingency_probability,
            alpha=args.alpha,
            loading_threshold=args.loading_threshold,
            severity_metric=args.severity_metric,
            pf_mode=args.risk_pf_mode,
            progress=args.progress,
            output_dir=args.output_dir,
        )
        contingency_count = len(probabilities)

        LOGGER.info(
            "Highest-risk operating point:\n%s",
            risk.risk_by_time.sort_values("total_risk", ascending=False)
            .head(1)
            .to_string(index=False),
        )
        LOGGER.info(
            "Highest-risk contingencies:\n%s",
            risk.risk_by_contingency.head(10).to_string(index=False),
        )

    if args.map_mode == "base":
        map_states = base_states
    else:
        if analysis is None:
            raise ValueError(
                f"--map-mode {args.map_mode!r} requires contingency analysis; "
                "remove --skip-risk."
            )
        if args.risk_pf_mode == "dc":
            LOGGER.warning(
                "The selected contingency map uses DC analysis. Bus-voltage "
                "magnitudes are therefore not a meaningful AC voltage-risk indicator."
            )
        map_states = contingency_map_states(
            net,
            analysis,
            probabilities,
            mode=args.map_mode,
            hours=args.hours,
        )

    figure_stem = (
        "24_hours_simulation_example"
        if args.map_mode == "base"
        else f"24_hours_{args.map_mode}_contingency_state"
    )
    output_png = args.output_dir / f"{figure_stem}.png"
    output_pdf = args.output_dir / f"{figure_stem}.pdf"

    LOGGER.info("Creating 24-panel network figure.")
    plot_24_hour_states(
        net,
        map_states,
        output_png=output_png,
        output_pdf=output_pdf,
        line_loading_vmax=args.line_loading_vmax,
        voltage_vmin=args.voltage_vmin,
        voltage_vmax=args.voltage_vmax,
        show_hour_titles=args.show_hour_titles,
    )

    write_run_metadata(
        args.output_dir / "run_metadata.json",
        args=args,
        net=net,
        profiles=profiles,
        contingency_count=contingency_count,
    )

    LOGGER.info("Saved figure: %s", output_png)
    LOGGER.info("Saved outputs under: %s", args.output_dir)


if __name__ == "__main__":
    main()
