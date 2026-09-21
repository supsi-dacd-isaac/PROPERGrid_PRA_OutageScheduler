"""Data adapters and diagnostics for nodal power-system time series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from .base import FloatArray, as_2d_float


Orientation = Literal["auto", "time-by-nodes", "nodes-by-time"]


@dataclass(frozen=True)
class NodalDataset:
    values: FloatArray
    node_names: tuple[str, ...]
    source: str
    variable: str | None
    original_shape: tuple[int, int]
    orientation_applied: str


@dataclass(frozen=True)
class NodalDiagnostics:
    n_observations: int
    n_nodes: int
    n_constant_nodes: int
    constant_node_indices: tuple[int, ...]
    n_active_nodes: int
    effective_rank: int
    rank_fraction: float
    min_active_spearman: float | None
    max_active_spearman: float | None
    fraction_zero: float
    fraction_negative: float
    warning: str | None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _resolve_input(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = [
        item
        for suffix in ("*.mat", "*.csv", "*.npy", "*.npz")
        for item in path.rglob(suffix)
        if item.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No .mat, .csv, .npy, or .npz file found under {path}.")
    preferred_tokens = ("hourlydemandbus", "nodal", "demand", "load")
    candidates.sort(
        key=lambda item: (
            not any(token in item.name.lower() for token in preferred_tokens),
            -item.stat().st_size,
            str(item),
        )
    )
    return candidates[0]


def _select_mat_variable(path: Path, variable: str | None) -> tuple[str, FloatArray]:
    variables = {name: (shape, kind) for name, shape, kind in whosmat(path)}
    if variable is not None:
        if variable not in variables:
            raise KeyError(f"Variable {variable!r} not found. Available: {sorted(variables)}")
        value = loadmat(path, variable_names=[variable])[variable]
        return variable, as_2d_float(value, name=variable)

    candidates = [
        (name, shape)
        for name, (shape, kind) in variables.items()
        if len(shape) == 2 and min(shape) >= 1 and kind in {"double", "single", "int64", "int32"}
    ]
    if not candidates:
        raise ValueError(f"No numeric two-dimensional array found in {path}.")
    candidates.sort(key=lambda item: (-(item[1][0] * item[1][1]), item[0]))
    selected = candidates[0][0]
    value = loadmat(path, variable_names=[selected])[selected]
    return selected, as_2d_float(value, name=selected)


def _orient(
    values: FloatArray,
    orientation: Orientation,
    expected_nodes: int | None,
) -> tuple[FloatArray, str]:
    if orientation == "time-by-nodes":
        return values, orientation
    if orientation == "nodes-by-time":
        return values.T, orientation
    if expected_nodes is not None:
        row_match = values.shape[0] == expected_nodes
        col_match = values.shape[1] == expected_nodes
        if row_match != col_match:
            return (values.T, "nodes-by-time (auto)") if row_match else (values, "time-by-nodes (auto)")
    rows, columns = values.shape
    if rows <= 1000 and columns >= 2 * rows:
        return values.T, "nodes-by-time (auto)"
    if columns <= 1000 and rows >= 2 * columns:
        return values, "time-by-nodes (auto)"
    raise ValueError(
        f"Ambiguous matrix orientation for shape {values.shape}. Supply --orientation or --expected-nodes."
    )


def load_nodal_data(
    path: str | Path,
    *,
    variable: str | None = None,
    orientation: Orientation = "auto",
    expected_nodes: int | None = None,
    time_column: str | None = None,
    node_column: str | None = None,
    value_column: str | None = None,
) -> NodalDataset:
    """Load wide/long CSV, MAT, NPY, or NPZ nodal data.

    Directories are searched recursively, preferring filenames containing
    ``hourlyDemandBus``, ``nodal``, ``demand``, or ``load``. For a Swiss long
    table, specify all of ``time_column``, ``node_column``, and ``value_column``.
    """

    resolved = _resolve_input(Path(path))
    selected_variable: str | None = None
    names: tuple[str, ...] | None = None
    suffix = resolved.suffix.lower()
    if suffix == ".mat":
        selected_variable, raw = _select_mat_variable(resolved, variable)
    elif suffix == ".npy":
        raw = as_2d_float(np.load(resolved), name=resolved.name)
    elif suffix == ".npz":
        archive = np.load(resolved)
        key = variable or max(archive.files, key=lambda item: np.asarray(archive[item]).size)
        raw = as_2d_float(archive[key], name=key)
        selected_variable = key
    elif suffix == ".csv":
        frame = pd.read_csv(resolved)
        long_fields = (time_column, node_column, value_column)
        if any(field is not None for field in long_fields):
            if not all(field is not None for field in long_fields):
                raise ValueError("Long CSV input requires time_column, node_column, and value_column together.")
            wide = frame.pivot_table(index=time_column, columns=node_column, values=value_column, aggfunc="mean")
            wide = wide.sort_index()
            names = tuple(map(str, wide.columns))
            raw = as_2d_float(wide.to_numpy(), name=resolved.name)
            orientation = "time-by-nodes"
        else:
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.empty:
                raise ValueError(f"No numeric columns found in {resolved}.")
            names = tuple(map(str, numeric.columns))
            raw = as_2d_float(numeric.to_numpy(), name=resolved.name)
            orientation = "time-by-nodes" if orientation == "auto" else orientation
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    original_shape = tuple(map(int, raw.shape))
    values, applied = _orient(raw, orientation, expected_nodes)
    if expected_nodes is not None and values.shape[1] != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} nodes but oriented data contain {values.shape[1]}.")
    if names is None or len(names) != values.shape[1]:
        names = tuple(f"bus_{idx + 1}" for idx in range(values.shape[1]))
    return NodalDataset(
        values=np.asarray(values, dtype=float),
        node_names=names,
        source=str(resolved),
        variable=selected_variable,
        original_shape=original_shape,
        orientation_applied=applied,
    )


def diagnose_nodal_data(values: FloatArray, *, tolerance: float = 1e-10) -> NodalDiagnostics:
    """Quantify degeneracy before fitting multivariate dependence models."""

    data = as_2d_float(values)
    standard_deviation = np.std(data, axis=0, ddof=1)
    active = standard_deviation > tolerance
    active_data = data[:, active]
    if active_data.shape[1] == 0:
        rank = 0
        min_corr = max_corr = None
    else:
        standardized = (active_data - active_data.mean(axis=0)) / active_data.std(axis=0, ddof=1)
        singular_values = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)
        rank = int(np.sum(singular_values > tolerance * singular_values[0])) if singular_values.size else 0
        if active_data.shape[1] > 1:
            ranked = pd.DataFrame(active_data).rank(method="average").to_numpy()
            correlation = np.corrcoef(ranked, rowvar=False)
            off_diagonal = correlation[np.triu_indices_from(correlation, 1)]
            min_corr = float(np.min(off_diagonal))
            max_corr = float(np.max(off_diagonal))
        else:
            min_corr = max_corr = None
    fraction = rank / max(int(active.sum()), 1)
    warning = None
    if active.sum() > 1 and fraction < 0.5:
        warning = (
            "The active nodal matrix is low rank; copula-family comparisons cannot be interpreted "
            "as evidence about general tail dependence without richer nodal observations."
        )
    return NodalDiagnostics(
        n_observations=int(data.shape[0]),
        n_nodes=int(data.shape[1]),
        n_constant_nodes=int((~active).sum()),
        constant_node_indices=tuple(np.flatnonzero(~active).astype(int).tolist()),
        n_active_nodes=int(active.sum()),
        effective_rank=rank,
        rank_fraction=float(fraction),
        min_active_spearman=min_corr,
        max_active_spearman=max_corr,
        fraction_zero=float(np.mean(data == 0.0)),
        fraction_negative=float(np.mean(data < 0.0)),
        warning=warning,
    )

