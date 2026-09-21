#!/usr/bin/env python3
"""Print and export a compact excerpt of IEEE nodal load data.

The script searches ``data/powersystems/<SYSTEM>`` for the nodal demand file,
orients it as time x bus, selects a small number of active buses, and writes a
report-ready CSV and LaTeX table to ``outputs/load_data_preview``.

Examples
--------
Run for both systems with the defaults (24 hours and six buses)::

    python probabilistic_model/print_load_data_for_report.py

Show 48 hours starting at hourly index 1000 for IEEE118::

    python probabilistic_model/print_load_data_for_report.py \
        --systems IEEE118 --start 1000 --hours 48 --nodes 8

Notes
-----
Pickle files must come from a trusted source: Python pickle deserialization can
execute code. The project data files are assumed to be trusted local inputs.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".pkl", ".pickle", ".parquet", ".csv", ".npy", ".npz"}


def project_root() -> Path:
    """Infer the repository root when this file is in probabilistic_model/."""
    script = Path(__file__).resolve()
    candidates = (script.parent.parent, Path.cwd(), Path.cwd().parent)
    for candidate in candidates:
        if (candidate / "data" / "powersystems").is_dir():
            return candidate
    return script.parent.parent


def expected_node_count(system: str) -> int | None:
    match = re.search(r"(\d+)", system)
    return int(match.group(1)) if match else None


def file_priority(path: Path) -> tuple[int, str]:
    """Prefer the conventional hourly-demand file over unrelated arrays."""
    name = path.name.lower()
    score = 0
    if "hourlydemandbus" in name:
        score += 100
    if "nodal" in name and ("load" in name or "demand" in name):
        score += 50
    if "load" in name or "demand" in name:
        score += 20
    if path.suffix.lower() in {".pkl", ".pickle"}:
        score += 10
    return score, name


def discover_demand_file(system_directory: Path) -> Path:
    candidates = [
        path
        for path in system_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No supported nodal-demand file found below {system_directory}"
        )
    return max(candidates, key=file_priority)


def read_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        print(f"Warning: loading trusted pickle file {path}", file=sys.stderr)
        with path.open("rb") as stream:
            return pickle.load(stream)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        arrays = [archive[key] for key in archive.files if archive[key].ndim == 2]
        if not arrays:
            raise ValueError(f"No two-dimensional array found in {path}")
        return max(arrays, key=lambda value: value.size)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def two_dimensional_candidates(value: Any, depth: int = 0) -> Iterable[Any]:
    """Yield 2-D tables, including tables nested in a dict/list pickle."""
    if depth > 3:
        return
    if isinstance(value, pd.DataFrame):
        yield value
    elif isinstance(value, pd.Series):
        yield value.to_frame()
    elif isinstance(value, np.ndarray) and value.ndim == 2:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from two_dimensional_candidates(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from two_dimensional_candidates(item, depth + 1)


def make_unique(labels: Iterable[Any]) -> list[str]:
    """Make duplicate bus labels explicit rather than silently dropping them."""
    counts: dict[str, int] = {}
    unique: list[str] = []
    for value in labels:
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
        unique.append(label if counts[label] == 1 else f"{label} ({counts[label]})")
    return unique


def orient_as_time_by_bus(raw: Any, system: str) -> pd.DataFrame:
    candidates = list(two_dimensional_candidates(raw))
    if not candidates:
        raise ValueError(f"No two-dimensional load table found in {type(raw)!r}")

    largest = max(candidates, key=lambda value: int(np.prod(value.shape)))
    expected = expected_node_count(system)

    if isinstance(largest, pd.DataFrame):
        frame = largest.copy()
        rows, columns = frame.shape
        transpose = (
            expected is not None and rows == expected and columns != expected
        ) or (
            not (expected is not None and columns == expected) and rows < columns
        )
        if transpose:
            frame = frame.T
    else:
        values = np.asarray(largest)
        rows, columns = values.shape
        transpose = (
            expected is not None and rows == expected and columns != expected
        ) or (
            not (expected is not None and columns == expected) and rows < columns
        )
        if transpose:
            values = values.T
        frame = pd.DataFrame(
            values,
            columns=[f"bus_{index + 1}" for index in range(values.shape[1])],
        )

    # Convert only after orientation so original bus labels are retained.
    frame.columns = make_unique(frame.columns)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(axis=0, limit_direction="both").ffill().bfill()

    if frame.empty or frame.isna().any().any():
        raise ValueError("The selected demand table is empty or has unresolved values")
    if frame.shape[0] <= frame.shape[1]:
        raise ValueError(f"Expected time x bus data; obtained shape {frame.shape}")
    return frame.astype(float)


def select_report_excerpt(
    loads: pd.DataFrame,
    start: int,
    hours: int,
    nodes: int,
    selection: str,
    zero_tolerance: float,
) -> tuple[pd.DataFrame, int]:
    if start < 0 or hours < 1 or nodes < 1:
        raise ValueError("start must be >= 0; hours and nodes must be >= 1")
    if start >= len(loads):
        raise IndexError(f"start={start} is outside the {len(loads)}-hour dataset")

    mean_absolute_load = loads.abs().mean(axis=0)
    active = mean_absolute_load[mean_absolute_load > zero_tolerance]
    if active.empty:
        raise ValueError("No active demand buses were detected")

    if selection == "busiest":
        selected = active.sort_values(ascending=False).index[:nodes]
    else:
        selected = active.index[:nodes]

    stop = min(start + hours, len(loads))
    excerpt = loads.iloc[start:stop].loc[:, selected].copy()
    # Total includes every bus, including active buses omitted from the table.
    excerpt["Total system load"] = loads.iloc[start:stop].sum(axis=1)

    if isinstance(excerpt.index, pd.DatetimeIndex):
        excerpt.index.name = "Timestamp"
    else:
        excerpt.index = pd.Index(range(start, stop), name="Hour index")
    return excerpt, len(active)


def safe_label(system: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", system).strip("_").lower()


def process_system(args: argparse.Namespace, system: str) -> None:
    system_directory = args.data_root / system
    if not system_directory.is_dir():
        case_insensitive = [
            path
            for path in args.data_root.iterdir()
            if path.is_dir() and path.name.lower() == system.lower()
        ] if args.data_root.is_dir() else []
        if not case_insensitive:
            raise FileNotFoundError(f"System directory not found: {system_directory}")
        system_directory = case_insensitive[0]

    source = discover_demand_file(system_directory)
    loads = orient_as_time_by_bus(read_file(source), system)
    excerpt, active_count = select_report_excerpt( loads, start=args.start, hours=args.hours,  nodes=args.nodes,
                                                   selection=args.selection, zero_tolerance=args.zero_tolerance,)
    print("\n" + "=" * 78)
    print(f"{system}: nodal load excerpt")
    print(f"Source: {source}")
    print(f"Dataset: {len(loads):,} hourly rows x {loads.shape[1]} buses")
    print(f"Active demand buses: {active_count}")
    print(f"Displayed hours: {excerpt.index[0]} to {excerpt.index[-1]}")
    print(f"Displayed buses: {', '.join(excerpt.columns[:-1])}")
    print("=" * 78)
    print(excerpt.round(args.decimals).to_string())

    if args.no_export:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_label(system)}_load_excerpt_h{args.start}_{len(excerpt)}h"
    csv_path = args.output_dir / f"{stem}.csv"
    tex_path = args.output_dir / f"{stem}.tex"
    excerpt.round(args.decimals).to_csv(csv_path)

    caption = (
        f"Illustrative nodal and total load excerpt for {system}. "
        f"The displayed buses are selected using the '{args.selection}' rule."
    )
    latex = excerpt.reset_index().to_latex(
        index=False,
        float_format=lambda value: f"{value:.{args.decimals}f}",
        caption=caption,
        label=f"tab:{safe_label(system)}_load_excerpt",
        position="htbp",
    )
    tex_path.write_text(latex, encoding="utf-8")
    print(f"\nSaved CSV:   {csv_path}")
    print(f"Saved LaTeX: {tex_path}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=["IEEE24", "IEEE118"],  help="System directory names below data/powersystems (default: both).", )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "data" / "powersystems",
        help="Directory containing the IEEE24 and IEEE118 folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "load_data_preview",
        help="Destination for report-ready CSV and LaTeX tables.",
    )
    parser.add_argument("--start", type=int, default=0, help="First hourly row.")
    parser.add_argument("--hours", type=int, default=24, help="Rows to display.")
    parser.add_argument("--nodes", type=int, default=6, help="Bus columns to display.")
    parser.add_argument(
        "--selection",
        choices=("busiest", "first"),
        default="busiest",
        help="How to choose the displayed active buses.",
    )
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1e-10,
        help="Mean absolute load below this threshold denotes an inactive bus.",
    )
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Print only; do not write CSV or LaTeX files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for system in args.systems:
        process_system(args, system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
