"""Data loading utilities for PROPER PRA/PSA experiments.

The expected project layout is::

    data/<system_name>/System.xlsx
    data/<system_name>/System.m
    data/<system_name>/hourlyDemandBus.pkl

The loader reads the MATPOWER-style system data and, when pandapower is
available, converts it into a pandapower network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import tempfile

import numpy as np
import pandas as pd


BUS_COLUMNS = [
    "bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va",
    "baseKV", "zone", "Vmax", "Vmin",
]
GEN_COLUMNS = [
    "bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin",
    "Pc1", "Pc2", "Qc1min", "Qc1max", "Qc2min", "Qc2max", "ramp_agc", "ramp_10",
    "ramp_30", "ramp_q", "apf",
]
BRANCH_COLUMNS = [
    "fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio", "angle",
    "status", "angmin", "angmax",
]
GENCOST_COLUMNS = ["model", "startup", "shutdown", "n", "c2", "c1", "c0"]


@dataclass
class SystemData:
    """Container returned by :func:`load_system`."""

    system_name: str
    system_dir: Path
    net: Any | None
    load_time_series: pd.DataFrame | None
    bus: pd.DataFrame
    gen: pd.DataFrame
    branch: pd.DataFrame
    gencost: pd.DataFrame | None
    base_mva: float
    metadata: dict[str, Any]


def load_system(
    data_dir: str | Path = "data",
    system_name: str = "IEEE24",
    *,
    as_pandapower: bool = True,
    f_hz: float = 50.0,
) -> SystemData:
    """Load a power-system test case from ``data/<system_name>``.

    Parameters
    ----------
    data_dir:
        Root data directory.
    system_name:
        Name of the system subfolder, e.g. ``IEEE24``.
    as_pandapower:
        If ``True``, convert the MATPOWER data to a pandapower network.
        If pandapower is not installed, a clear ImportError is raised.
    f_hz:
        Network frequency passed to pandapower conversion.
    """
    system_dir = Path(data_dir) / system_name
    if not system_dir.exists():
        raise FileNotFoundError(f"System folder not found: {system_dir}")

    xlsx_path = system_dir / "System.xlsx"
    m_path = system_dir / "System.m"
    if xlsx_path.exists():
        case = read_matpower_xlsx(xlsx_path)
        source = str(xlsx_path)
    elif m_path.exists():
        case = read_matpower_m(m_path)
        source = str(m_path)
    else:
        raise FileNotFoundError(f"No System.xlsx or System.m found in {system_dir}")

    load_ts = load_hourly_demand(system_dir)
    if load_ts is not None:
        load_ts = normalize_bus_load_columns(load_ts, case["bus"])

    net = None
    if as_pandapower:
        net = matpower_case_to_pandapower(case, f_hz=f_hz)

    metadata = {
        "source": source,
        "n_bus": len(case["bus"]),
        "n_gen": len(case["gen"]),
        "n_branch": len(case["branch"]),
        "load_time_series_shape": None if load_ts is None else load_ts.shape,
        "bus_ids": case["bus"]["bus_i"].astype(int).tolist(),
    }
    if net is not None:
        attach_matpower_metadata(net, case)

    return SystemData(
        system_name=system_name,
        system_dir=system_dir,
        net=net,
        load_time_series=load_ts,
        bus=case["bus"],
        gen=case["gen"],
        branch=case["branch"],
        gencost=case.get("gencost"),
        base_mva=float(case["baseMVA"]),
        metadata=metadata,
    )


def read_matpower_xlsx(path: str | Path) -> dict[str, Any]:
    """Read a MATPOWER case stored as the provided Excel workbook."""
    path = Path(path)
    bus = pd.read_excel(path, sheet_name="bus", header=None)
    gen = pd.read_excel(path, sheet_name="gen", header=None)
    branch = pd.read_excel(path, sheet_name="branch", header=None)
    gencost = None
    try:
        gencost = pd.read_excel(path, sheet_name="gencost", header=None)
    except ValueError:
        pass
    base_mva = float(pd.read_excel(path, sheet_name="baseMVA", header=None).iloc[0, 0])
    return _case_from_raw_frames(base_mva, bus, gen, branch, gencost)


def read_matpower_m(path: str | Path) -> dict[str, Any]:
    """Parse the relevant matrices from a MATPOWER ``.m`` case file."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", text)
    if not base_match:
        raise ValueError(f"Could not find mpc.baseMVA in {path}")
    base_mva = float(base_match.group(1))
    bus = pd.DataFrame(_parse_matpower_matrix(text, "bus"))
    gen = pd.DataFrame(_parse_matpower_matrix(text, "gen"))
    branch = pd.DataFrame(_parse_matpower_matrix(text, "branch"))
    try:
        gencost = pd.DataFrame(_parse_matpower_matrix(text, "gencost"))
    except ValueError:
        gencost = None
    return _case_from_raw_frames(base_mva, bus, gen, branch, gencost)


def _case_from_raw_frames(
    base_mva: float,
    bus: pd.DataFrame,
    gen: pd.DataFrame,
    branch: pd.DataFrame,
    gencost: pd.DataFrame | None,
) -> dict[str, Any]:
    bus = bus.iloc[:, : len(BUS_COLUMNS)].copy()
    gen = gen.iloc[:, : len(GEN_COLUMNS)].copy()
    branch = branch.iloc[:, : len(BRANCH_COLUMNS)].copy()
    bus.columns = BUS_COLUMNS
    gen.columns = GEN_COLUMNS
    branch.columns = BRANCH_COLUMNS
    for col in ["bus_i", "type", "area", "zone"]:
        bus[col] = bus[col].astype(int)
    for col in ["bus", "status"]:
        gen[col] = gen[col].astype(int)
    for col in ["fbus", "tbus", "status"]:
        branch[col] = branch[col].astype(int)
    if gencost is not None:
        gencost = gencost.copy()
        if gencost.shape[1] >= len(GENCOST_COLUMNS):
            gencost = gencost.iloc[:, : len(GENCOST_COLUMNS)]
            gencost.columns = GENCOST_COLUMNS
    return {"baseMVA": float(base_mva), "bus": bus, "gen": gen, "branch": branch, "gencost": gencost}


def _parse_matpower_matrix(text: str, name: str) -> list[list[float]]:
    pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find mpc.{name} matrix")
    body = match.group(1)
    rows: list[list[float]] = []
    for raw_line in body.splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue
        line = line.rstrip(";").strip()
        if not line:
            continue
        values = [float(v) for v in re.split(r"\s+", line) if v]
        if values:
            rows.append(values)
    if not rows:
        raise ValueError(f"mpc.{name} matrix is empty")
    return rows


def matpower_case_to_pandapower(case: dict[str, Any], *, f_hz: float = 50.0):
    """Convert parsed MATPOWER case data to pandapower.

    The conversion is routed through pandapower's ``from_ppc`` converter. This
    keeps the implementation compact and consistent with pandapower's internal
    MATPOWER conventions.
    """
    try:
        import pandapower.converter as pc
    except ImportError as exc:
        raise ImportError(
            "pandapower is required for as_pandapower=True. Install it with "
            "`pip install pandapower`."
        ) from exc

    ppc = {
        "version": "2",
        "baseMVA": float(case["baseMVA"]),
        "bus": case["bus"][BUS_COLUMNS].to_numpy(dtype=float),
        "gen": case["gen"][GEN_COLUMNS].to_numpy(dtype=float),
        "branch": case["branch"][BRANCH_COLUMNS].to_numpy(dtype=float),
    }
    if case.get("gencost") is not None:
        ppc["gencost"] = case["gencost"].to_numpy(dtype=float)
    net = pc.from_ppc(ppc, f_hz=f_hz)
    attach_matpower_metadata(net, case)
    return net


def load_hourly_demand(system_dir: str | Path) -> pd.DataFrame | None:
    """Load hourly bus demand if present."""
    system_dir = Path(system_dir)
    pkl_path = system_dir / "hourlyDemandBus.pkl"
    mat_path = system_dir / "hourlyDemandBus.mat"
    xlsx_path = system_dir / "System.xlsx"
    if pkl_path.exists():
        obj = pd.read_pickle(pkl_path)
        return pd.DataFrame(obj).copy()
    if mat_path.exists():
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        candidates = [v for k, v in mat.items() if not k.startswith("__") and np.ndim(v) == 2]
        if candidates:
            return pd.DataFrame(candidates[0])
    if xlsx_path.exists():
        try:
            return pd.read_excel(xlsx_path, sheet_name="hourlyDemandBus", header=None)
        except ValueError:
            return None
    return None


def normalize_bus_load_columns(load_ts: pd.DataFrame, bus: pd.DataFrame) -> pd.DataFrame:
    """Normalize load time-series columns to MATPOWER external bus IDs.

    Input files may use ``bus_0`` ... ``bus_23``. The returned dataframe uses
    the corresponding external MATPOWER bus IDs, e.g. 1 ... 24.
    """
    out = load_ts.copy()
    bus_ids = bus["bus_i"].astype(int).tolist()
    if out.shape[1] != len(bus_ids):
        return out
    out.columns = bus_ids
    return out


def attach_matpower_metadata(net: Any, case: dict[str, Any]) -> None:
    """Attach external MATPOWER IDs to a pandapower network in-place."""
    bus_ids = case["bus"]["bus_i"].astype(int).tolist()
    if hasattr(net, "bus") and len(net.bus) == len(bus_ids):
        # Pandapower conversion may keep buses in MATPOWER row order. Store the
        # explicit external IDs to avoid ambiguity when applying load profiles.
        net.bus["matpower_bus_id"] = bus_ids
    if hasattr(net, "line") and len(net.line) + len(getattr(net, "trafo", [])) <= len(case["branch"]):
        net.line["matpower_branch_source"] = np.arange(len(net.line))
