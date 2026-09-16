#!/usr/bin/env python3
"""
Benchmark probabilistic forecasts of nodal hourly demand on IEEE test systems.

The script is intentionally self-contained. It loads the nodal demand matrix from

    data/powersystems/IEEE24
    data/powersystems/IEEE118

when it is placed either in the repository root or in probabilistic_model/. Results
are written by default to output/nodal_forecasting.

Implemented forecasters
-----------------------
1. Seasonal empirical baseline (required for skill scores).
2. Regularized multivariate autoregression with independent Gaussian innovations.
3. Gaussian copula fitted pairwise to autoregressive innovations.
4. Student-t copula fitted pairwise to autoregressive innovations.
5. Tree-structured pairwise Gaussian-mixture copula model (GMCM approximation).
6. Empirical copula using a moving-block bootstrap of innovation vectors.
7. Low-rank spatio-temporal multi-output Gaussian process (LMC/PCA approximation).
8. Conditional empirical copula with beta kernels and a Schaake shuffle.

The conditional empirical copula follows the beta-kernel construction in:
P. F. Austnes et al., Sustainable Energy, Grids and Networks 42 (2025), 101708.
The Gaussian-process model is the homogeneous continuous-output specialization of
the LMC idea in P. Moreno-Munoz et al., NeurIPS 2018. It is deliberately low-rank
so that IEEE118 remains feasible on a workstation.

Install
-------
python -m pip install numpy pandas scipy scikit-learn matplotlib joblib

Examples
--------
Quick smoke test:
python probabilistic_model/benchmark_nodal_forecasters.py --profile quick

Standard benchmark:
python probabilistic_model/benchmark_nodal_forecasters.py --systems IEEE24 IEEE118

More accurate Monte Carlo evaluation:
python probabilistic_model/benchmark_nodal_forecasters.py --profile full --save-models

Notes
-----
* Splits are chronological. Hyperparameters are selected before the test period.
* Forecasts are evaluated at rolling origins for 1, 24 and 168 hours.
* AIC/BIC are reported only where a finite-dimensional likelihood or copula
  pseudo-likelihood is available. They must not be compared across different
  criterion_scope values.
* The primary ranking is out-of-sample and combines normalized MAE, CRPS, energy
  score, calibration error, system-aggregate CRPS and Wasserstein distance.
* Pickle files must be trusted: loading arbitrary pickle files can execute code.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import platform
import sys
import traceback
import warnings
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import gammaln, logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import norm, rankdata, t as student_t
from scipy.stats import wasserstein_distance
from sklearn import __version__ as sklearn_version
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture

LOGGER = logging.getLogger("nodal_forecast_benchmark")
EPS = 1.0e-10
MODEL_NAMES = ( "seasonal_empirical",
                "autoregressive_gaussian",
                "gaussian_copula",
                "student_t_copula",
                "gmcm",
                "empirical_copula",
                "spatiotemporal_gp",
                "conditional_empirical_copula", )


@dataclass
class Config:
    systems: Tuple[str, ...] = ("IEEE24", "IEEE118")
    horizons: Tuple[int, ...] = (1, 24, 168)
    models: Tuple[str, ...] = MODEL_NAMES
    n_scenarios: int = 200
    max_origins: int = 20
    seed: int = 20260914
    train_fraction: float = 0.65
    validation_end_fraction: float = 0.80
    lags: Tuple[int, ...] = (1, 24, 168)
    ridge_alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    gmcm_max_components: int = 3
    gp_rank: int = 8
    gp_max_train: int = 400
    conditional_bandwidth: float = 0.10
    max_variogram_pairs: int = 250
    save_models: bool = False
    make_plots: bool = True


@dataclass
class Dataset:
    values: np.ndarray
    node_names: List[str]
    timestamps: Optional[pd.DatetimeIndex]
    source_path: Path


def stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int((base + zlib.crc32(payload)) % (2**32 - 1))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (here, here.parent, Path.cwd(), Path.cwd().parent)
    for candidate in candidates:
        if (candidate / "data" / "powersystems").exists():
            return candidate
    # Expected placement is project_root/probabilistic_model/this_script.py.
    return here.parent.parent


def expected_nodes(system: str) -> Optional[int]:
    digits = "".join(c for c in system if c.isdigit())
    return int(digits) if digits else None


def discover_system_directory(data_root: Path, system: str) -> Path:
    exact = data_root / system
    if exact.is_dir():
        return exact
    matches = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower() == system.lower()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"System directory not found: {exact}")


def file_priority(path: Path) -> Tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "hourlydemandbus" in name:
        score += 100
    if "nodal" in name and "demand" in name:
        score += 50
    if "demand" in name or "load" in name:
        score += 20
    suffix_score = {
        ".pkl": 10,
        ".pickle": 10,
        ".parquet": 8,
        ".csv": 6,
        ".npz": 5,
        ".npy": 4,
    }.get(path.suffix.lower(), 0)
    return score + suffix_score, -len(path.parts), name


def discover_data_file(system_dir: Path) -> Path:
    suffixes = {".pkl", ".pickle", ".parquet", ".csv", ".npz", ".npy"}
    candidates = [p for p in system_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    if not candidates:
        raise FileNotFoundError(f"No supported demand file found below {system_dir}")
    candidates.sort(key=file_priority, reverse=True)
    selected = candidates[0]
    LOGGER.info("Selected demand file: %s", selected)
    return selected


def extract_2d_object(obj: Any) -> Any:
    candidates: List[Any] = []

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 3:
            return
        if isinstance(value, pd.DataFrame):
            if value.ndim == 2:
                candidates.append(value)
        elif isinstance(value, pd.Series):
            candidates.append(value.to_frame())
        elif isinstance(value, np.ndarray):
            if value.ndim == 2:
                candidates.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item, depth + 1)

    visit(obj)
    if not candidates:
        raise ValueError(f"No two-dimensional array or DataFrame found in object of type {type(obj)!r}")
    return max(candidates, key=lambda x: int(np.prod(x.shape)))


def read_candidate(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        LOGGER.warning("Loading trusted pickle file %s", path)
        with path.open("rb") as stream:
            return pickle.load(stream)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if frame.shape[1] > 1:
            first = frame.iloc[:, 0]
            first_name = str(frame.columns[0]).lower()
            looks_temporal = (
                not pd.api.types.is_numeric_dtype(first)
                or "date" in first_name
                or "time" in first_name
            )
            if looks_temporal:
                parsed = pd.to_datetime(first, errors="coerce", utc=False)
                if parsed.notna().mean() > 0.95:
                    frame = frame.iloc[:, 1:].copy()
                    frame.index = pd.DatetimeIndex(parsed)
        return frame
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        arrays = [archive[key] for key in archive.files if archive[key].ndim == 2]
        if not arrays:
            raise ValueError(f"No two-dimensional array in {path}")
        return max(arrays, key=lambda x: x.size)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def orient_and_clean(raw: Any, system: str) -> Tuple[np.ndarray, List[str], Optional[pd.DatetimeIndex]]:
    obj = extract_2d_object(raw)
    n_expected = expected_nodes(system)

    if isinstance(obj, pd.DataFrame):
        frame = obj.copy()
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            numeric = frame.apply(pd.to_numeric, errors="coerce")
        frame = numeric
        rows, cols = frame.shape

        transpose = False
        if n_expected is not None and rows == n_expected and cols != n_expected:
            transpose = True
        elif n_expected is not None and cols == n_expected:
            transpose = False
        elif rows < cols and rows <= max(2 * (n_expected or rows), 256):
            transpose = True

        if transpose:
            original_index = frame.index
            frame = frame.T
            node_names = [str(x) for x in original_index]
        else:
            node_names = [str(x) for x in frame.columns]

        timestamps = None
        if isinstance(frame.index, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(frame.index)
        elif not pd.api.types.is_numeric_dtype(frame.index.dtype):
            parsed = pd.to_datetime(frame.index, errors="coerce")
            if len(parsed) and pd.Series(parsed).notna().mean() > 0.95:
                timestamps = pd.DatetimeIndex(parsed)

        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(axis=1, how="all")
        node_names = [str(x) for x in frame.columns]
        frame = frame.interpolate(axis=0, limit_direction="both").ffill().bfill()
        values = frame.to_numpy(dtype=float)
    else:
        values = np.asarray(obj, dtype=float)
        rows, cols = values.shape
        if n_expected is not None and rows == n_expected and cols != n_expected:
            values = values.T
        elif n_expected is not None and cols == n_expected:
            pass
        elif rows < cols:
            values = values.T
        node_names = [f"node_{i + 1}" for i in range(values.shape[1])]
        timestamps = None
        values[~np.isfinite(values)] = np.nan
        values = pd.DataFrame(values).interpolate(limit_direction="both").ffill().bfill().to_numpy()

    if values.ndim != 2 or values.shape[0] <= values.shape[1]:
        raise ValueError(f"Expected time x node matrix; obtained {values.shape}")
    if np.isnan(values).any():
        bad = np.flatnonzero(np.isnan(values).any(axis=0))
        raise ValueError(f"Unresolved missing values in nodes {bad[:10].tolist()}")
    if values.shape[0] < 4 * 168:
        raise ValueError(
            f"Only {values.shape[0]} hourly samples found. At least {4 * 168} are required "
            "for a meaningful weekly benchmark."
        )
    if timestamps is not None:
        timestamps = timestamps[: values.shape[0]]
    return values, node_names, timestamps


def load_dataset(data_root: Path, system: str) -> Dataset:
    system_dir = discover_system_directory(data_root, system)
    path = discover_data_file(system_dir)
    values, node_names, timestamps = orient_and_clean(read_candidate(path), system)
    LOGGER.info(
        "%s: %d hourly observations, %d nodal series, source=%s",
        system,
        values.shape[0],
        values.shape[1],
        path,
    )
    return Dataset(values, node_names, timestamps, path)


def time_components(length: int, timestamps: Optional[pd.DatetimeIndex]) -> Tuple[np.ndarray, np.ndarray]:
    if timestamps is not None and len(timestamps) >= length:
        hour = timestamps.hour.to_numpy(dtype=int)
        weekend = (timestamps.dayofweek.to_numpy(dtype=int) >= 5).astype(int)
    else:
        idx = np.arange(length)
        hour = idx % 24
        weekend = ((idx // 24) % 7 >= 5).astype(int)
    return hour, weekend


def calendar_features(length: int, timestamps: Optional[pd.DatetimeIndex]) -> np.ndarray:
    idx = np.arange(length, dtype=float)
    if timestamps is not None and len(timestamps) >= length:
        hour = timestamps.hour.to_numpy(dtype=float)
        hour_week = timestamps.dayofweek.to_numpy(dtype=float) * 24.0 + hour
        day_year = timestamps.dayofyear.to_numpy(dtype=float)
        weekend = (timestamps.dayofweek.to_numpy(dtype=float) >= 5.0).astype(float)
    else:
        hour = idx % 24.0
        hour_week = idx % 168.0
        day_year = (idx / 24.0) % 365.2425
        weekend = (((idx // 24.0) % 7.0) >= 5.0).astype(float)
    trend = 2.0 * idx / max(length - 1, 1) - 1.0
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * hour_week / 168.0),
            np.cos(2 * np.pi * hour_week / 168.0),
            np.sin(2 * np.pi * day_year / 365.2425),
            np.cos(2 * np.pi * day_year / 365.2425),
            weekend,
            trend,
        ]
    )


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    for j in range(x.shape[1]):
        result[:, j] = rankdata(x[:, j], method="average") / (x.shape[0] + 1.0)
    return np.clip(result, 1.0e-6, 1.0 - 1.0e-6)


def nearest_correlation(matrix: np.ndarray, shrink: float = 0.02) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 0:
        return np.ones((1, 1))
    matrix = np.nan_to_num((matrix + matrix.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 1.0)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1.0e-6)
    matrix = (vectors * values) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(matrix), EPS))
    matrix = matrix / np.outer(scale, scale)
    matrix = (1.0 - shrink) * matrix + shrink * np.eye(matrix.shape[0])
    np.fill_diagonal(matrix, 1.0)
    return matrix


def empirical_inverse(sorted_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    sorted_values = np.asarray(sorted_values)
    u = np.asarray(u)
    n, d = sorted_values.shape
    grid = (np.arange(n, dtype=float) + 0.5) / n
    out = np.empty_like(u, dtype=float)
    for j in range(d):
        out[:, j] = np.interp(u[:, j], grid, sorted_values[:, j])
    return out


def make_origins(test_start: int, n_time: int, horizon: int, max_origins: int) -> np.ndarray:
    candidates = np.arange(test_start, n_time - horizon + 1, max(horizon, 1), dtype=int)
    if candidates.size == 0:
        return candidates
    if max_origins > 0 and candidates.size > max_origins:
        indices = np.unique(np.linspace(0, candidates.size - 1, max_origins).round().astype(int))
        candidates = candidates[indices]
    return candidates


class RidgeBackbone:
    """Regularized VARX-style one-step conditional mean model."""

    def __init__(self, lags: Sequence[int], alphas: Sequence[float]):
        self.lags = tuple(sorted(set(int(x) for x in lags)))
        self.alphas = tuple(float(x) for x in alphas)

    def _design(self, y: np.ndarray, times: np.ndarray) -> np.ndarray:
        parts = [(y[times - lag] - self.y_mean) / self.y_scale for lag in self.lags]
        parts.append(self.calendar[times])
        return np.concatenate(parts, axis=1)

    def fit(self, y: np.ndarray, train_end: int, fit_end: int, calendar: np.ndarray) -> "RidgeBackbone":
        self.calendar = calendar
        self.y_mean = y[:train_end].mean(axis=0)
        self.y_scale = y[:train_end].std(axis=0)
        self.y_scale = np.where(self.y_scale > EPS, self.y_scale, 1.0)
        start = max(self.lags)
        train_times = np.arange(start, train_end)
        validation_times = np.arange(train_end, fit_end)
        x_train = self._design(y, train_times)
        target_train = (y[train_times] - self.y_mean) / self.y_scale

        best_alpha = self.alphas[0]
        best_score = np.inf
        for alpha in self.alphas:
            candidate = Ridge(alpha=alpha, fit_intercept=True)
            candidate.fit(x_train, target_train)
            if validation_times.size:
                prediction = candidate.predict(self._design(y, validation_times))
                truth = (y[validation_times] - self.y_mean) / self.y_scale
                score = float(np.mean(np.abs(prediction - truth)))
            else:
                score = float(np.mean(np.abs(candidate.predict(x_train) - target_train)))
            if score < best_score:
                best_alpha, best_score = alpha, score

        fit_times = np.arange(start, fit_end)
        self.model = Ridge(alpha=best_alpha, fit_intercept=True)
        self.model.fit(self._design(y, fit_times), (y[fit_times] - self.y_mean) / self.y_scale)
        fitted = self.model.predict(self._design(y, fit_times)) * self.y_scale + self.y_mean
        self.residuals = y[fit_times] - fitted
        self.residual_times = fit_times
        self.alpha = float(best_alpha)
        self.validation_nmae = float(best_score)
        self.n_nodes = y.shape[1]
        self.n_features = self._design(y, fit_times[:1]).shape[1]
        return self

    def predict_from_lags(self, lag_values: Sequence[np.ndarray], absolute_time: int) -> np.ndarray:
        n_rows = lag_values[0].shape[0]
        parts = [(value - self.y_mean) / self.y_scale for value in lag_values]
        parts.append(np.repeat(self.calendar[absolute_time][None, :], n_rows, axis=0))
        x = np.concatenate(parts, axis=1)
        return self.model.predict(x) * self.y_scale + self.y_mean


class IndependentGaussianInnovation:
    def fit(self, residuals: np.ndarray) -> "IndependentGaussianInnovation":
        self.location = residuals.mean(axis=0)
        self.scale = residuals.std(axis=0, ddof=1)
        floor = max(float(np.nanmedian(self.scale[self.scale > EPS])) * 1.0e-4, 1.0e-8)
        self.scale = np.maximum(self.scale, floor)
        self.loglik = float(np.sum(norm.logpdf(residuals, loc=self.location, scale=self.scale)))
        self.n_obs = int(residuals.size)
        self.n_parameters = int(2 * residuals.shape[1])
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.location, self.scale, size=(n, self.location.size))


class GaussianCopulaInnovation:
    def fit(self, residuals: np.ndarray) -> "GaussianCopulaInnovation":
        scales = residuals.std(axis=0)
        floor = max(float(np.nanmedian(scales[scales > EPS])) * 1.0e-6, 1.0e-10)
        self.active = np.flatnonzero(scales > floor)
        self.n_total = residuals.shape[1]
        self.constant = residuals.mean(axis=0)
        if self.active.size == 0:
            self.correlation = np.ones((1, 1))
            self.sorted = np.empty((residuals.shape[0], 0))
            self.loglik = 0.0
            self.n_parameters = 0
            self.n_obs = residuals.shape[0]
            return self
        active_residuals = residuals[:, self.active]
        self.sorted = np.sort(active_residuals, axis=0)
        u = pseudo_observations(active_residuals)
        z = norm.ppf(u)
        if z.shape[1] == 1:
            corr = np.ones((1, 1))
        else:
            corr = np.corrcoef(z, rowvar=False)
        self.correlation = nearest_correlation(corr)
        self.cholesky = np.linalg.cholesky(self.correlation)
        sign, logdet = np.linalg.slogdet(self.correlation)
        inverse = np.linalg.inv(self.correlation)
        quad = np.einsum("ij,jk,ik->i", z, inverse - np.eye(z.shape[1]), z)
        self.loglik = float(np.sum(-0.5 * logdet - 0.5 * quad)) if sign > 0 else np.nan
        d = self.active.size
        self.n_parameters = int(d * (d - 1) // 2)
        self.n_obs = int(residuals.shape[0])
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        result = np.repeat(self.constant[None, :], n, axis=0)
        if self.active.size:
            z = rng.standard_normal((n, self.active.size)) @ self.cholesky.T
            result[:, self.active] = empirical_inverse(self.sorted, norm.cdf(z))
        return result


def t_copula_loglik(z: np.ndarray, correlation: np.ndarray, df: float) -> float:
    n, d = z.shape
    if d <= 1:
        return 0.0
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0:
        return -np.inf
    inverse = np.linalg.inv(correlation)
    quad = np.einsum("ij,jk,ik->i", z, inverse, z)
    joint = (
        gammaln((df + d) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * (d * np.log(df * np.pi) + logdet)
        - 0.5 * (df + d) * np.log1p(quad / df)
    )
    marginal = np.sum(student_t.logpdf(z, df=df), axis=1)
    return float(np.sum(joint - marginal))


class StudentTCopulaInnovation:
    def __init__(self, df_grid: Sequence[float] = (3, 4, 5, 8, 12, 20, 30)):
        self.df_grid = tuple(float(x) for x in df_grid)

    def fit(self, residuals: np.ndarray) -> "StudentTCopulaInnovation":
        scales = residuals.std(axis=0)
        floor = max(float(np.nanmedian(scales[scales > EPS])) * 1.0e-6, 1.0e-10)
        self.active = np.flatnonzero(scales > floor)
        self.n_total = residuals.shape[1]
        self.constant = residuals.mean(axis=0)
        self.n_obs = int(residuals.shape[0])
        if self.active.size == 0:
            self.df = 30.0
            self.correlation = np.ones((1, 1))
            self.sorted = np.empty((residuals.shape[0], 0))
            self.loglik = 0.0
            self.n_parameters = 1
            return self
        active_residuals = residuals[:, self.active]
        self.sorted = np.sort(active_residuals, axis=0)
        u = pseudo_observations(active_residuals)
        # Select the common degrees of freedom on at most 3,000 evenly spaced
        # innovations; then refit and score the selected model on all data.
        selection = np.unique(
            np.linspace(0, u.shape[0] - 1, min(3000, u.shape[0])).round().astype(int)
        )
        best_df, best_selection_loglik = None, -np.inf
        for df in self.df_grid:
            z_selection = student_t.ppf(u[selection], df=df)
            corr_selection = (
                np.ones((1, 1))
                if z_selection.shape[1] == 1
                else np.corrcoef(z_selection, rowvar=False)
            )
            corr_selection = nearest_correlation(corr_selection)
            ll = t_copula_loglik(z_selection, corr_selection, df)
            if ll > best_selection_loglik:
                best_df, best_selection_loglik = df, ll
        self.df = float(best_df)
        z = student_t.ppf(u, df=self.df)
        corr = np.ones((1, 1)) if z.shape[1] == 1 else np.corrcoef(z, rowvar=False)
        self.correlation = nearest_correlation(corr)
        self.loglik = t_copula_loglik(z, self.correlation, self.df)
        self.cholesky = np.linalg.cholesky(self.correlation)
        d = self.active.size
        self.n_parameters = int(d * (d - 1) // 2 + 1)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        result = np.repeat(self.constant[None, :], n, axis=0)
        if self.active.size:
            normal_draw = rng.standard_normal((n, self.active.size)) @ self.cholesky.T
            scale = np.sqrt(rng.chisquare(self.df, size=n) / self.df)
            z = normal_draw / scale[:, None]
            result[:, self.active] = empirical_inverse(self.sorted, student_t.cdf(z, df=self.df))
        return result


def maximum_spanning_tree(correlation: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
    d = correlation.shape[0]
    if d <= 1:
        return 0, []
    weight = np.abs(correlation).copy()
    np.fill_diagonal(weight, -np.inf)
    root = int(np.argmax(np.sum(np.maximum(weight, 0.0), axis=1)))
    selected = np.zeros(d, dtype=bool)
    selected[root] = True
    edges: List[Tuple[int, int]] = []
    for _ in range(d - 1):
        best_weight, best_parent, best_child = -np.inf, -1, -1
        parents = np.flatnonzero(selected)
        children = np.flatnonzero(~selected)
        for parent in parents:
            local = children[np.argmax(weight[parent, children])]
            if weight[parent, local] > best_weight:
                best_weight = weight[parent, local]
                best_parent, best_child = int(parent), int(local)
        if best_child < 0:
            best_parent = int(parents[0])
            best_child = int(children[0])
        selected[best_child] = True
        edges.append((best_parent, best_child))
    return root, edges


class GMCMTreeInnovation:
    """
    Pairwise GMCM approximation assembled as a valid first-tree factorization.

    Each tree edge is a Gaussian mixture fitted in normal-score space. Conditional
    sampling along the maximum-spanning tree produces a coherent joint scenario.
    """

    def __init__(self, max_components: int = 3, seed: int = 0):
        self.max_components = int(max_components)
        self.seed = int(seed)

    def fit(self, residuals: np.ndarray) -> "GMCMTreeInnovation":
        scales = residuals.std(axis=0)
        floor = max(float(np.nanmedian(scales[scales > EPS])) * 1.0e-6, 1.0e-10)
        self.active = np.flatnonzero(scales > floor)
        self.n_total = residuals.shape[1]
        self.constant = residuals.mean(axis=0)
        self.n_obs = int(residuals.shape[0])
        if self.active.size == 0:
            self.root, self.edges, self.gmms = 0, [], {}
            self.sorted = np.empty((residuals.shape[0], 0))
            self.loglik, self.n_parameters = 0.0, 0
            return self
        active_residuals = residuals[:, self.active]
        self.sorted = np.sort(active_residuals, axis=0)
        u = pseudo_observations(active_residuals)
        z = norm.ppf(u)
        corr = np.ones((1, 1)) if z.shape[1] == 1 else nearest_correlation(np.corrcoef(z, rowvar=False))
        self.root, self.edges = maximum_spanning_tree(corr)
        self.gmms: Dict[int, GaussianMixture] = {}
        total_loglik = 0.0
        total_parameters = 0
        for edge_number, (parent, child) in enumerate(self.edges):
            pair = z[:, [parent, child]]
            best_model, best_bic = None, np.inf
            max_components = min(self.max_components, max(1, pair.shape[0] // 40))
            for components in range(1, max_components + 1):
                model = GaussianMixture(
                    n_components=components,
                    covariance_type="full",
                    reg_covar=1.0e-5,
                    n_init=2,
                    max_iter=300,
                    random_state=self.seed + edge_number * 17 + components,
                )
                model.fit(pair)
                bic = model.bic(pair)
                if bic < best_bic:
                    best_model, best_bic = model, bic
            assert best_model is not None
            self.gmms[child] = best_model
            edge_ll = best_model.score_samples(pair) - norm.logpdf(pair[:, 0]) - norm.logpdf(pair[:, 1])
            total_loglik += float(np.sum(edge_ll))
            k = best_model.n_components
            total_parameters += int((k - 1) + 2 * k + 3 * k)
        self.loglik = float(total_loglik)
        self.n_parameters = int(total_parameters)
        return self

    @staticmethod
    def _conditional_draw(
        parent_value: np.ndarray, model: GaussianMixture, rng: np.random.Generator
    ) -> np.ndarray:
        means = model.means_
        covariance = model.covariances_
        var_parent = np.maximum(covariance[:, 0, 0], EPS)
        log_weight = (
            np.log(np.maximum(model.weights_, EPS))[None, :]
            - 0.5 * np.log(2.0 * np.pi * var_parent)[None, :]
            - 0.5 * (parent_value[:, None] - means[:, 0][None, :]) ** 2 / var_parent[None, :]
        )
        probabilities = np.exp(log_weight - logsumexp(log_weight, axis=1)[:, None])
        cumulative = np.cumsum(probabilities, axis=1)
        component = np.sum(rng.random(parent_value.size)[:, None] > cumulative, axis=1)
        component = np.minimum(component, model.n_components - 1)
        cov_cp = covariance[component, 1, 0]
        var_p = np.maximum(covariance[component, 0, 0], EPS)
        conditional_mean = means[component, 1] + cov_cp / var_p * (
            parent_value - means[component, 0]
        )
        conditional_variance = covariance[component, 1, 1] - cov_cp**2 / var_p
        return conditional_mean + np.sqrt(np.maximum(conditional_variance, 1.0e-8)) * rng.standard_normal(
            parent_value.size
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        result = np.repeat(self.constant[None, :], n, axis=0)
        if self.active.size == 0:
            return result
        z = np.zeros((n, self.active.size))
        z[:, self.root] = rng.standard_normal(n)
        for parent, child in self.edges:
            z[:, child] = self._conditional_draw(z[:, parent], self.gmms[child], rng)
        result[:, self.active] = empirical_inverse(self.sorted, norm.cdf(z))
        return result


class EmpiricalBlockInnovation:
    def fit(self, residuals: np.ndarray) -> "EmpiricalBlockInnovation":
        self.residuals = np.asarray(residuals)
        self.loglik = np.nan
        self.n_parameters = np.nan
        self.n_obs = residuals.shape[0]
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        index = rng.integers(0, self.residuals.shape[0], size=n)
        return self.residuals[index]

    def sample_path(self, n: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        available = self.residuals.shape[0] - horizon + 1
        if available <= 0:
            index = rng.integers(0, self.residuals.shape[0], size=(n, horizon))
            return self.residuals[index]
        starts = rng.integers(0, available, size=n)
        return self.residuals[starts[:, None] + np.arange(horizon)[None, :]]


class RidgeInnovationForecaster:
    def __init__(
        self,
        name: str,
        backbone: RidgeBackbone,
        innovation: Any,
        criterion_scope: str,
    ):
        self.name = name
        self.backbone = backbone
        self.innovation = innovation
        ll = float(getattr(innovation, "loglik", np.nan))
        k = float(getattr(innovation, "n_parameters", np.nan))
        n = int(getattr(innovation, "n_obs", backbone.residuals.shape[0]))
        if name == "autoregressive_gaussian" and np.isfinite(k):
            k += backbone.n_nodes * (backbone.n_features + 1)
        self.fit_info = {
            "model": name,
            "fit_loglik": ll,
            "n_parameters": k,
            "n_likelihood_observations": n,
            "aic": 2.0 * k - 2.0 * ll if np.isfinite(ll) and np.isfinite(k) else np.nan,
            "bic": np.log(max(n, 1)) * k - 2.0 * ll if np.isfinite(ll) and np.isfinite(k) else np.nan,
            "criterion_scope": criterion_scope,
            "selected_ridge_alpha": backbone.alpha,
            "validation_nmae_standardized": backbone.validation_nmae,
        }
        if hasattr(innovation, "df"):
            self.fit_info["student_t_df"] = innovation.df

    def simulate(
        self,
        y: np.ndarray,
        origin: int,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_nodes = y.shape[1]
        simulations = np.empty((n_scenarios, horizon, n_nodes), dtype=float)
        innovation_path = None
        if hasattr(self.innovation, "sample_path"):
            innovation_path = self.innovation.sample_path(n_scenarios, horizon, rng)
        for lead in range(horizon):
            absolute_time = origin + lead
            lag_values = []
            for lag in self.backbone.lags:
                source_time = absolute_time - lag
                if source_time < origin:
                    lag_values.append(np.repeat(y[source_time][None, :], n_scenarios, axis=0))
                else:
                    lag_values.append(simulations[:, source_time - origin, :])
            conditional_mean = self.backbone.predict_from_lags(lag_values, absolute_time)
            innovations = (
                innovation_path[:, lead, :]
                if innovation_path is not None
                else self.innovation.sample(n_scenarios, rng)
            )
            # Do not clip at zero: the PROPER nodal series can be net load,
            # where negative values denote net injection into the grid.
            simulations[:, lead, :] = conditional_mean + innovations
        return simulations


class SeasonalEmpiricalForecaster:
    name = "seasonal_empirical"

    def __init__(self, lag: int = 168):
        self.lag = int(lag)

    def fit(self, y: np.ndarray, fit_end: int) -> "SeasonalEmpiricalForecaster":
        lag = self.lag if fit_end > self.lag + 24 else 24
        self.lag = lag
        residuals = y[lag:fit_end] - y[: fit_end - lag]
        self.innovation = EmpiricalBlockInnovation().fit(residuals)
        self.fit_info = {
            "model": self.name,
            "fit_loglik": np.nan,
            "n_parameters": np.nan,
            "n_likelihood_observations": residuals.shape[0],
            "aic": np.nan,
            "bic": np.nan,
            "criterion_scope": "not_available_nonparametric",
            "seasonal_lag": lag,
        }
        return self

    def simulate(
        self,
        y: np.ndarray,
        origin: int,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        errors = self.innovation.sample_path(n_scenarios, horizon, rng)
        base = np.stack([y[origin + lead - self.lag] for lead in range(horizon)])
        return base[None, :, :] + errors


class SpatioTemporalGPForecaster:
    name = "spatiotemporal_gp"

    def __init__(self, rank: int, max_train: int, seed: int):
        self.rank = int(rank)
        self.max_train = int(max_train)
        self.seed = int(seed)

    def fit(
        self,
        y: np.ndarray,
        fit_end: int,
        calendar: np.ndarray,
    ) -> "SpatioTemporalGPForecaster":
        self.y_mean = y[:fit_end].mean(axis=0)
        self.y_scale = y[:fit_end].std(axis=0)
        self.y_scale = np.where(self.y_scale > EPS, self.y_scale, 1.0)
        standardized = (y[:fit_end] - self.y_mean) / self.y_scale
        rank = min(self.rank, standardized.shape[1], max(1, standardized.shape[0] - 1))
        self.pca = PCA(n_components=rank, svd_solver="randomized", random_state=self.seed)
        factors = self.pca.fit_transform(standardized)
        reconstructed = self.pca.inverse_transform(factors)
        self.node_noise = np.maximum(np.std(standardized - reconstructed, axis=0), 1.0e-3)
        self.calendar = calendar
        # A weekly latent-state lag is known for every lead up to 168 hours.
        # Conditioning on it makes the GP a genuine forecaster rather than a
        # calendar-only smoother, while preserving leakage-free multi-step use.
        self.conditioning_lag = 168 if fit_end > 168 + 48 else 24
        eligible = np.arange(self.conditioning_lag, fit_end)
        design = np.concatenate(
            [calendar[eligible], factors[eligible - self.conditioning_lag]], axis=1
        )
        self.x_mean = design.mean(axis=0)
        self.x_scale = np.where(design.std(axis=0) > EPS, design.std(axis=0), 1.0)
        positions = np.unique(
            np.linspace(0, eligible.size - 1, min(self.max_train, eligible.size))
            .round()
            .astype(int)
        )
        subset = eligible[positions]
        x = (design[positions] - self.x_mean) / self.x_scale
        self.gps: List[GaussianProcessRegressor] = []
        log_marginal = 0.0
        for component in range(rank):
            kernel = (
                ConstantKernel(1.0, (0.05, 20.0))
                * RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0))
                + WhiteKernel(noise_level=0.05, noise_level_bounds=(1.0e-5, 2.0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1.0e-6,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=self.seed + component,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(x, factors[subset, component])
            self.gps.append(gp)
            log_marginal += float(gp.log_marginal_likelihood_value_)
        self.fit_info = {
            "model": self.name,
            "fit_loglik": log_marginal,
            "n_parameters": np.nan,
            "n_likelihood_observations": len(subset),
            "aic": np.nan,
            "bic": np.nan,
            "criterion_scope": "gp_log_marginal_likelihood_not_aic_comparable",
            "gp_rank": rank,
            "gp_training_points": len(subset),
            "conditioning_lag": self.conditioning_lag,
            "explained_variance_ratio": float(np.sum(self.pca.explained_variance_ratio_)),
        }
        return self

    def simulate(
        self,
        y: np.ndarray,
        origin: int,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        future_times = np.arange(origin, origin + horizon)
        lagged_standardized = (
            y[future_times - self.conditioning_lag] - self.y_mean
        ) / self.y_scale
        lagged_factors = self.pca.transform(lagged_standardized)
        x_future_raw = np.concatenate(
            [self.calendar[future_times], lagged_factors], axis=1
        )
        x_future = (x_future_raw - self.x_mean) / self.x_scale
        factor_draws = np.empty((n_scenarios, horizon, len(self.gps)))
        for component, gp in enumerate(self.gps):
            mean, covariance = gp.predict(x_future, return_cov=True)
            covariance = (covariance + covariance.T) / 2.0 + 1.0e-7 * np.eye(horizon)
            values, vectors = np.linalg.eigh(covariance)
            root = vectors @ np.diag(np.sqrt(np.maximum(values, 1.0e-9)))
            factor_draws[:, :, component] = mean[None, :] + rng.standard_normal(
                (n_scenarios, horizon)
            ) @ root.T
        standardized = np.einsum("shr,rn->shn", factor_draws, self.pca.components_)
        standardized += self.pca.mean_[None, None, :]
        standardized += rng.normal(
            0.0,
            self.node_noise[None, None, :],
            size=(n_scenarios, horizon, self.y_mean.size),
        )
        return standardized * self.y_scale + self.y_mean


class ConditionalEmpiricalCopulaForecaster:
    """
    Nodewise beta-kernel conditional empirical copula plus Schaake shuffle.

    For every node and lead time, the target is conditioned on its value one week
    earlier. The one-week lag is observed for all requested horizons up to 168 h.
    Nodewise conditional samples are reordered with historical multivariate blocks,
    thereby restoring empirical spatial and temporal rank dependence.
    """

    name = "conditional_empirical_copula"

    def __init__(self, bandwidth: float = 0.10, lag: int = 168):
        self.bandwidth = float(bandwidth)
        self.lag = int(lag)

    def fit(
        self,
        y: np.ndarray,
        fit_end: int,
        hours: np.ndarray,
        weekend: np.ndarray,
    ) -> "ConditionalEmpiricalCopulaForecaster":
        self.training = y[:fit_end]
        self.fit_end = int(fit_end)
        # Retain the complete calendar for test-origin lookup; only demand values
        # before fit_end enter the fitted conditional distributions.
        self.hours = np.asarray(hours)
        self.weekend = np.asarray(weekend)
        self.cache: Dict[Tuple[int, int], np.ndarray] = {}
        eligible = np.arange(self.lag, fit_end)
        for hour in range(24):
            for day_class in (0, 1):
                idx = eligible[(self.hours[eligible] == hour) & (self.weekend[eligible] == day_class)]
                if idx.size < 20:
                    idx = eligible[self.hours[eligible] == hour]
                if idx.size < 20:
                    idx = eligible
                self.cache[(hour, day_class)] = idx
        self.fit_info = {
            "model": self.name,
            "fit_loglik": np.nan,
            "n_parameters": np.nan,
            "n_likelihood_observations": fit_end - self.lag,
            "aic": np.nan,
            "bic": np.nan,
            "criterion_scope": "not_available_beta_kernel_nonparametric",
            "bandwidth": self.bandwidth,
            "conditioning_lag": self.lag,
            "joint_reordering": "Schaake_shuffle_historical_blocks",
        }
        return self

    @staticmethod
    def _rank_query(sample: np.ndarray, query: float) -> float:
        sorted_sample = np.sort(sample)
        rank = np.searchsorted(sorted_sample, query, side="right")
        return float(np.clip((rank + 0.5) / (sample.size + 1.0), 1.0e-4, 1.0 - 1.0e-4))

    def _conditional_node_samples(
        self,
        node: int,
        candidate_times: np.ndarray,
        query_lag: float,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        target = self.training[candidate_times, node]
        conditioning = self.training[candidate_times - self.lag, node]
        if np.std(target) <= EPS:
            return np.full(n_scenarios, float(np.mean(target)))
        u_target = rankdata(target, method="average") / (target.size + 1.0)
        u_conditioning = rankdata(conditioning, method="average") / (conditioning.size + 1.0)
        q = self._rank_query(conditioning, query_lag)
        h = self.bandwidth
        a = q / h + 1.0
        b = (1.0 - q) / h + 1.0
        log_weight = beta_dist.logpdf(np.clip(u_conditioning, 1.0e-8, 1.0 - 1.0e-8), a, b)
        if not np.isfinite(log_weight).any():
            probabilities = np.full(target.size, 1.0 / target.size)
        else:
            log_weight -= logsumexp(log_weight)
            probabilities = np.exp(log_weight)
        selected = rng.choice(target.size, size=n_scenarios, replace=True, p=probabilities)
        centers = np.clip(u_target[selected], 1.0e-5, 1.0 - 1.0e-5)
        u_draw = rng.beta(centers / h + 1.0, (1.0 - centers) / h + 1.0)
        target_sorted = np.sort(target)
        grid = (np.arange(target.size, dtype=float) + 0.5) / target.size
        return np.interp(u_draw, grid, target_sorted)

    def simulate(
        self,
        y: np.ndarray,
        origin: int,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_nodes = y.shape[1]
        raw = np.empty((n_scenarios, horizon, n_nodes))
        for lead in range(horizon):
            absolute_time = origin + lead
            hour = absolute_time % 24
            if absolute_time < len(self.hours):
                hour = int(self.hours[absolute_time])
            day_class = int(self.weekend[absolute_time]) if absolute_time < len(self.weekend) else 0
            candidate_times = self.cache[(hour, day_class)]
            lag_value = y[absolute_time - self.lag]
            for node in range(n_nodes):
                raw[:, lead, node] = self._conditional_node_samples(
                    node, candidate_times, lag_value[node], n_scenarios, rng
                )

        max_start = self.fit_end - horizon
        starts = np.arange(self.lag, max_start + 1)
        if starts.size == 0:
            starts = np.arange(max(0, self.fit_end - horizon), self.fit_end)
        matching = starts[self.hours[starts] == self.hours[origin]]
        if matching.size >= 5:
            starts = matching
        chosen = rng.choice(starts, size=n_scenarios, replace=True)
        template = self.training[chosen[:, None] + np.arange(horizon)[None, :]]

        scenarios = np.empty_like(raw)
        for lead in range(horizon):
            for node in range(n_nodes):
                order = np.argsort(template[:, lead, node], kind="mergesort")
                scenarios[order, lead, node] = np.sort(raw[:, lead, node])
        return scenarios


def ensemble_crps(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS for ensemble axis 0; remaining dimensions are preserved."""
    ensemble = np.asarray(ensemble)
    truth = np.asarray(truth)
    n = ensemble.shape[0]
    first = np.mean(np.abs(ensemble - truth[None, ...]), axis=0)
    ordered = np.sort(ensemble, axis=0)
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    second = np.tensordot(weights, ordered, axes=(0, 0)) / (n * n)
    return first - second


def pinball_loss(truth: np.ndarray, quantile_forecast: np.ndarray, level: float) -> np.ndarray:
    error = truth - quantile_forecast
    return np.maximum(level * error, (level - 1.0) * error)


def energy_score(
    ensemble: np.ndarray,
    truth: np.ndarray,
    node_scale: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = 256,
) -> float:
    standardized_ensemble = ensemble / node_scale[None, None, :]
    standardized_truth = truth / node_scale[None, :]
    values = []
    for lead in range(truth.shape[0]):
        draws = standardized_ensemble[:, lead, :]
        target = standardized_truth[lead]
        first = np.mean(np.linalg.norm(draws - target[None, :], axis=1))
        a = rng.integers(0, draws.shape[0], size=n_pairs)
        b = rng.integers(0, draws.shape[0], size=n_pairs)
        second = 0.5 * np.mean(np.linalg.norm(draws[a] - draws[b], axis=1))
        values.append(first - second)
    return float(np.mean(values))


def variogram_score(
    ensemble: np.ndarray,
    truth: np.ndarray,
    pairs: np.ndarray,
    node_scale: np.ndarray,
    order: float = 0.5,
) -> float:
    if pairs.size == 0:
        return np.nan
    ens = ensemble / node_scale[None, None, :]
    obs = truth / node_scale[None, :]
    i, j = pairs[:, 0], pairs[:, 1]
    observed = np.abs(obs[:, i] - obs[:, j]) ** order
    predicted = np.mean(np.abs(ens[:, :, i] - ens[:, :, j]) ** order, axis=0)
    return float(np.mean((predicted - observed) ** 2))


def select_variogram_pairs(y: np.ndarray, maximum: int) -> np.ndarray:
    active = np.flatnonzero(np.std(y, axis=0) > EPS)
    if active.size < 2:
        return np.empty((0, 2), dtype=int)
    corr = np.corrcoef(y[:, active], rowvar=False)
    upper = np.triu_indices(active.size, k=1)
    weights = np.abs(np.nan_to_num(corr[upper]))
    take = min(maximum, weights.size)
    selected = np.argpartition(weights, -take)[-take:] if take < weights.size else np.arange(weights.size)
    return np.column_stack([active[upper[0][selected]], active[upper[1][selected]]])


def evaluate_forecaster(
    forecaster: Any,
    system: str,
    y: np.ndarray,
    node_names: Sequence[str],
    timestamps: Optional[pd.DatetimeIndex],
    origins: np.ndarray,
    horizon: int,
    n_scenarios: int,
    node_scale: np.ndarray,
    variogram_pairs: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    quantile_levels = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    truth_chunks, median_chunks, mean_chunks = [], [], []
    crps_chunks, draw_chunks, quantile_chunks = [], [], []
    aggregate_truth_chunks, aggregate_median_chunks = [], []
    aggregate_crps_chunks, aggregate_quantile_chunks = [], []
    energy_values, variogram_values, nll_values = [], [], []
    example_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for origin_number, origin in enumerate(origins):
        ensemble = forecaster.simulate(y, int(origin), horizon, n_scenarios, rng)
        if ensemble.shape != (n_scenarios, horizon, y.shape[1]):
            raise ValueError(f"{forecaster.name} returned invalid shape {ensemble.shape}")
        if not np.isfinite(ensemble).all():
            raise FloatingPointError(f"{forecaster.name} returned non-finite scenarios")
        truth = y[origin : origin + horizon]
        quantiles = np.quantile(ensemble, quantile_levels, axis=0)
        median = quantiles[3]
        mean = np.mean(ensemble, axis=0)
        crps = ensemble_crps(ensemble, truth)
        # A small pooled predictive sample makes the marginal Wasserstein
        # diagnostic substantially less sensitive to one Monte Carlo draw.
        n_wasserstein_draws = min(10, n_scenarios)
        chosen = rng.integers(0, n_scenarios, size=(n_wasserstein_draws, horizon))
        predictive_draw = ensemble[
            chosen, np.arange(horizon)[None, :], :
        ].reshape(-1, y.shape[1])

        truth_chunks.append(truth)
        median_chunks.append(median)
        mean_chunks.append(mean)
        crps_chunks.append(crps)
        draw_chunks.append(predictive_draw)
        quantile_chunks.append(quantiles)

        standard_deviation = np.maximum(np.std(ensemble, axis=0, ddof=1), 1.0e-8)
        nll = 0.5 * np.log(2.0 * np.pi * standard_deviation**2) + 0.5 * (
            (truth - mean) / standard_deviation
        ) ** 2
        nll_values.append(float(np.mean(nll)))
        energy_values.append(energy_score(ensemble, truth, node_scale, rng))
        variogram_values.append(variogram_score(ensemble, truth, variogram_pairs, node_scale))

        aggregate_ensemble = ensemble.sum(axis=2)
        aggregate_truth = truth.sum(axis=1)
        aggregate_quantiles = np.quantile(aggregate_ensemble, quantile_levels, axis=0)
        aggregate_truth_chunks.append(aggregate_truth)
        aggregate_median_chunks.append(aggregate_quantiles[3])
        aggregate_crps_chunks.append(ensemble_crps(aggregate_ensemble, aggregate_truth))
        aggregate_quantile_chunks.append(aggregate_quantiles)

        if origin_number == 0:
            for lead in range(horizon):
                timestamp = (
                    str(timestamps[origin + lead])
                    if timestamps is not None and origin + lead < len(timestamps)
                    else str(origin + lead)
                )
                example_rows.append(
                    {
                        "system": system,
                        "horizon": horizon,
                        "model": forecaster.name,
                        "origin": int(origin),
                        "lead": lead + 1,
                        "timestamp": timestamp,
                        "actual_total": aggregate_truth[lead],
                        "q05_total": aggregate_quantiles[0, lead],
                        "q10_total": aggregate_quantiles[1, lead],
                        "q50_total": aggregate_quantiles[3, lead],
                        "q90_total": aggregate_quantiles[5, lead],
                        "q95_total": aggregate_quantiles[6, lead],
                        "mean_total": float(np.mean(aggregate_ensemble[:, lead])),
                    }
                )

    truth = np.concatenate(truth_chunks, axis=0)
    median = np.concatenate(median_chunks, axis=0)
    mean = np.concatenate(mean_chunks, axis=0)
    crps = np.concatenate(crps_chunks, axis=0)
    predictive_draw = np.concatenate(draw_chunks, axis=0)
    quantiles = np.concatenate(quantile_chunks, axis=1)
    error = median - truth

    denominator_node = np.mean(np.abs(truth), axis=0)
    denominator_node = np.where(denominator_node > EPS, denominator_node, np.nan)
    mae_node = np.mean(np.abs(error), axis=0)
    rmse_node = np.sqrt(np.mean(error**2, axis=0))
    crps_node = np.mean(crps, axis=0)
    nmae_node = mae_node / denominator_node
    ncrps_node = crps_node / denominator_node
    coverage90_node = np.mean((truth >= quantiles[0]) & (truth <= quantiles[6]), axis=0)
    width90_node = np.mean(quantiles[6] - quantiles[0], axis=0)
    calibration_by_q_node = np.stack(
        [np.mean(truth <= quantiles[q], axis=0) - level for q, level in enumerate(quantile_levels)]
    )
    mace_node = np.mean(np.abs(calibration_by_q_node), axis=0)
    wasserstein_node = np.array(
        [wasserstein_distance(truth[:, j], predictive_draw[:, j]) for j in range(y.shape[1])]
    )
    normalized_wasserstein_node = wasserstein_node / denominator_node

    nodal = pd.DataFrame(
        {
            "system": system,
            "horizon": horizon,
            "model": forecaster.name,
            "node": list(node_names),
            "mae": mae_node,
            "nmae": nmae_node,
            "rmse": rmse_node,
            "crps": crps_node,
            "ncrps": ncrps_node,
            "coverage90": coverage90_node,
            "width90": width90_node,
            "mace": mace_node,
            "wasserstein": wasserstein_node,
            "normalized_wasserstein": normalized_wasserstein_node,
        }
    )
    valid = np.flatnonzero(np.isfinite(nmae_node))
    best = int(valid[np.argmin(nmae_node[valid])]) if valid.size else -1
    worst = int(valid[np.argmax(nmae_node[valid])]) if valid.size else -1

    aggregate_truth = np.concatenate(aggregate_truth_chunks)
    aggregate_median = np.concatenate(aggregate_median_chunks)
    aggregate_crps = np.concatenate(aggregate_crps_chunks)
    aggregate_quantiles = np.concatenate(aggregate_quantile_chunks, axis=1)
    aggregate_scale = max(float(np.mean(np.abs(aggregate_truth))), EPS)
    aggregate_error = aggregate_median - aggregate_truth
    aggregate_calibration = np.array(
        [
            np.mean(aggregate_truth <= aggregate_quantiles[q]) - level
            for q, level in enumerate(quantile_levels)
        ]
    )

    active_mask = np.isfinite(denominator_node)
    overall_scale = max(float(np.mean(np.abs(truth[:, active_mask]))), EPS)
    pinball = np.mean(
        [
            np.mean(pinball_loss(truth, quantiles[q], level))
            for q, level in enumerate(quantile_levels)
        ]
    )
    summary = {
        "system": system,
        "horizon": horizon,
        "model": forecaster.name,
        "n_origins": int(origins.size),
        "n_evaluated_hours": int(truth.shape[0]),
        "n_nodes": int(y.shape[1]),
        "mae": float(np.mean(np.abs(error[:, active_mask]))),
        "nmae": float(np.mean(np.abs(error[:, active_mask])) / overall_scale),
        "rmse": float(np.sqrt(np.mean(error[:, active_mask] ** 2))),
        "crps": float(np.mean(crps[:, active_mask])),
        "ncrps": float(np.mean(crps[:, active_mask]) / overall_scale),
        "mean_pinball": float(pinball),
        "mace": float(np.mean(np.abs(calibration_by_q_node[:, active_mask]))),
        "coverage80": float(
            np.mean(
                (truth[:, active_mask] >= quantiles[1][:, active_mask])
                & (truth[:, active_mask] <= quantiles[5][:, active_mask])
            )
        ),
        "coverage90": float(
            np.mean(
                (truth[:, active_mask] >= quantiles[0][:, active_mask])
                & (truth[:, active_mask] <= quantiles[6][:, active_mask])
            )
        ),
        "mean_width90": float(np.mean((quantiles[6] - quantiles[0])[:, active_mask])),
        "oos_marginal_gaussian_nll": float(np.mean(nll_values)),
        "oos_marginal_gaussian_log_score": float(-np.mean(nll_values)),
        "energy_score_standardized": float(np.mean(energy_values)),
        "variogram_score_standardized": float(np.nanmean(variogram_values)),
        "normalized_wasserstein": float(np.nanmean(normalized_wasserstein_node)),
        "aggregate_mae": float(np.mean(np.abs(aggregate_error))),
        "aggregate_nmae": float(np.mean(np.abs(aggregate_error)) / aggregate_scale),
        "aggregate_rmse": float(np.sqrt(np.mean(aggregate_error**2))),
        "aggregate_crps": float(np.mean(aggregate_crps)),
        "aggregate_ncrps": float(np.mean(aggregate_crps) / aggregate_scale),
        "aggregate_mace": float(np.mean(np.abs(aggregate_calibration))),
        "aggregate_coverage90": float(
            np.mean((aggregate_truth >= aggregate_quantiles[0]) & (aggregate_truth <= aggregate_quantiles[6]))
        ),
        "best_node": node_names[best] if best >= 0 else None,
        "best_node_nmae": float(nmae_node[best]) if best >= 0 else np.nan,
        "worst_node": node_names[worst] if worst >= 0 else None,
        "worst_node_nmae": float(nmae_node[worst]) if worst >= 0 else np.nan,
    }
    return summary, nodal, pd.DataFrame(example_rows)


def build_forecasters(
    config: Config,
    y: np.ndarray,
    train_end: int,
    fit_end: int,
    calendar: np.ndarray,
    hours: np.ndarray,
    weekend: np.ndarray,
    system: str,
) -> Dict[str, Any]:
    requested = set(config.models)
    models: Dict[str, Any] = {}

    if "seasonal_empirical" in requested:
        models["seasonal_empirical"] = SeasonalEmpiricalForecaster().fit(y, fit_end)

    ridge_names = {
        "autoregressive_gaussian",
        "gaussian_copula",
        "student_t_copula",
        "gmcm",
        "empirical_copula",
    }
    if requested & ridge_names:
        valid_lags = tuple(lag for lag in config.lags if lag < train_end - 48)
        if not valid_lags:
            valid_lags = (1,)
        backbone = RidgeBackbone(valid_lags, config.ridge_alphas).fit(
            y, train_end, fit_end, calendar
        )
        residuals = backbone.residuals

        if "autoregressive_gaussian" in requested:
            innovation = IndependentGaussianInnovation().fit(residuals)
            models["autoregressive_gaussian"] = RidgeInnovationForecaster(
                "autoregressive_gaussian",
                backbone,
                innovation,
                "full_conditional_diagonal_gaussian_likelihood",
            )
        if "gaussian_copula" in requested:
            innovation = GaussianCopulaInnovation().fit(residuals)
            models["gaussian_copula"] = RidgeInnovationForecaster(
                "gaussian_copula",
                backbone,
                innovation,
                "copula_only_semiparametric_pseudo_likelihood",
            )
        if "student_t_copula" in requested:
            innovation = StudentTCopulaInnovation().fit(residuals)
            models["student_t_copula"] = RidgeInnovationForecaster(
                "student_t_copula",
                backbone,
                innovation,
                "copula_only_semiparametric_pseudo_likelihood",
            )
        if "gmcm" in requested:
            innovation = GMCMTreeInnovation(
                max_components=config.gmcm_max_components,
                seed=stable_seed(config.seed, system, "gmcm_fit"),
            ).fit(residuals)
            models["gmcm"] = RidgeInnovationForecaster(
                "gmcm",
                backbone,
                innovation,
                "copula_only_semiparametric_pseudo_likelihood",
            )
        if "empirical_copula" in requested:
            innovation = EmpiricalBlockInnovation().fit(residuals)
            models["empirical_copula"] = RidgeInnovationForecaster(
                "empirical_copula",
                backbone,
                innovation,
                "not_available_nonparametric",
            )

    if "spatiotemporal_gp" in requested:
        models["spatiotemporal_gp"] = SpatioTemporalGPForecaster(
            config.gp_rank,
            config.gp_max_train,
            stable_seed(config.seed, system, "gp_fit"),
        ).fit(y, fit_end, calendar)

    if "conditional_empirical_copula" in requested:
        models["conditional_empirical_copula"] = ConditionalEmpiricalCopulaForecaster(
            config.conditional_bandwidth
        ).fit(y, fit_end, hours, weekend)

    return models


def add_skill_scores_and_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["mae_skill_vs_seasonal"] = np.nan
    result["crps_skill_vs_seasonal"] = np.nan
    rank_metrics = [
        "nmae",
        "ncrps",
        "energy_score_standardized",
        "mace",
        "aggregate_ncrps",
        "normalized_wasserstein",
    ]
    diagnostic_metrics = rank_metrics + [
        "rmse",
        "mean_pinball",
        "variogram_score_standardized",
        "oos_marginal_gaussian_nll",
        "aggregate_nmae",
        "aggregate_rmse",
        "aggregate_mace",
    ]
    for (system, horizon), index in result.groupby(["system", "horizon"]).groups.items():
        block = result.loc[index]
        baseline = block[block["model"] == "seasonal_empirical"]
        if not baseline.empty:
            baseline_mae = float(baseline.iloc[0]["mae"])
            baseline_crps = float(baseline.iloc[0]["crps"])
            if baseline_mae > EPS:
                result.loc[index, "mae_skill_vs_seasonal"] = 1.0 - block["mae"] / baseline_mae
            if baseline_crps > EPS:
                result.loc[index, "crps_skill_vs_seasonal"] = 1.0 - block["crps"] / baseline_crps
        rank_columns = []
        for metric in diagnostic_metrics:
            column = f"rank_{metric}"
            result.loc[index, column] = block[metric].rank(method="average", ascending=True)
            if metric in rank_metrics:
                rank_columns.append(column)
        result.loc[index, "forecast_rank_score"] = result.loc[index, rank_columns].mean(axis=1)
        result.loc[index, "forecast_rank"] = result.loc[index, "forecast_rank_score"].rank(
            method="min", ascending=True
        )
    return result


def rank_fit_information(fit_info: pd.DataFrame) -> pd.DataFrame:
    result = fit_info.copy()
    result["fit_loglik_rank_within_scope"] = np.nan
    result["aic_rank_within_scope"] = np.nan
    result["bic_rank_within_scope"] = np.nan
    valid = result[
        result["aic"].notna()
        & result["bic"].notna()
        & ~result["criterion_scope"].astype(str).str.startswith("not_available")
    ]
    for (_, scope), index in valid.groupby(["system", "criterion_scope"]).groups.items():
        result.loc[index, "fit_loglik_rank_within_scope"] = result.loc[
            index, "fit_loglik"
        ].rank(method="min", ascending=False)
        result.loc[index, "aic_rank_within_scope"] = result.loc[index, "aic"].rank(
            method="min", ascending=True
        )
        result.loc[index, "bic_rank_within_scope"] = result.loc[index, "bic"].rank(
            method="min", ascending=True
        )
    return result


def build_model_comparison(summary: pd.DataFrame, fit_info: pd.DataFrame) -> pd.DataFrame:
    """Join horizon-specific forecast performance to once-per-fit diagnostics."""
    if summary.empty:
        return pd.DataFrame()
    if fit_info.empty:
        return summary.copy()
    return summary.merge(fit_info, on=["system", "model"], how="left", validate="many_to_one")


def build_overall_ranking(summary: pd.DataFrame, fit_info: pd.DataFrame) -> pd.DataFrame:
    """Summarize the primary forecast ranking across systems and horizons."""
    if summary.empty:
        return pd.DataFrame()

    total_cases = int(summary[["system", "horizon"]].drop_duplicates().shape[0])
    overall = (
        summary.groupby("model", as_index=False)
        .agg(
            successful_cases=("forecast_rank", "count"),
            systems_evaluated=("system", "nunique"),
            horizons_evaluated=("horizon", "nunique"),
            forecast_wins=("forecast_rank", lambda values: int(np.sum(values == 1))),
            mean_forecast_rank=("forecast_rank", "mean"),
            mean_forecast_rank_score=("forecast_rank_score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ncrps=("ncrps", "mean"),
            mean_energy_score_standardized=("energy_score_standardized", "mean"),
            mean_mace=("mace", "mean"),
            mean_aggregate_ncrps=("aggregate_ncrps", "mean"),
            mean_normalized_wasserstein=("normalized_wasserstein", "mean"),
            mean_mae_skill_vs_seasonal=("mae_skill_vs_seasonal", "mean"),
            mean_crps_skill_vs_seasonal=("crps_skill_vs_seasonal", "mean"),
        )
    )
    overall["case_coverage_fraction"] = overall["successful_cases"] / max(total_cases, 1)

    if not fit_info.empty:
        comparable = (
            fit_info.groupby("model", as_index=False)
            .agg(
                comparable_aic_cases=("aic", lambda values: int(values.notna().sum())),
                mean_aic_rank_within_scope=("aic_rank_within_scope", "mean"),
                mean_bic_rank_within_scope=("bic_rank_within_scope", "mean"),
                mean_loglik_rank_within_scope=("fit_loglik_rank_within_scope", "mean"),
            )
        )
        overall = overall.merge(comparable, on="model", how="left", validate="one_to_one")

    # Penalize incomplete model coverage so a partially successful model cannot
    # win the overall table merely by being evaluated on easy cases.
    n_models = max(int(overall.shape[0]), 1)
    overall["coverage_adjusted_rank_score"] = overall["mean_forecast_rank_score"] + (
        1.0 - overall["case_coverage_fraction"]
    ) * n_models
    overall["overall_forecast_rank"] = overall["coverage_adjusted_rank_score"].rank(
        method="min", ascending=True
    )
    return overall.sort_values(
        ["overall_forecast_rank", "coverage_adjusted_rank_score", "model"]
    ).reset_index(drop=True)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def make_figures(summary: pd.DataFrame, examples: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for (system, horizon), block in summary.groupby(["system", "horizon"]):
        ordered = block.sort_values("forecast_rank_score")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(ordered))))
        ax.barh(ordered["model"], ordered["forecast_rank_score"], color="#3b82f6")
        ax.invert_yaxis()
        ax.set_xlabel("Mean out-of-sample rank (lower is better)")
        ax.set_title(f"{system}: {horizon}-hour probabilistic forecast")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{safe_name(system)}_H{horizon}_ranking.png", dpi=180)
        plt.close(fig)

        best_model = str(ordered.iloc[0]["model"])
        example = examples[
            (examples["system"] == system)
            & (examples["horizon"] == horizon)
            & (examples["model"] == best_model)
        ].sort_values("lead")
        if not example.empty:
            x = example["lead"].to_numpy()
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.fill_between(
                x,
                example["q10_total"].to_numpy(dtype=float),
                example["q90_total"].to_numpy(dtype=float),
                color="#93c5fd",
                alpha=0.55,
                label="Q10-Q90",
            )
            ax.plot(x, example["q50_total"], color="#1d4ed8", label="Predictive median")
            ax.plot(x, example["actual_total"], color="#111827", linewidth=1.5, label="Observed")
            ax.set_xlabel("Lead time [h]")
            ax.set_ylabel("Total nodal demand [input units]")
            ax.set_title(f"{system}: best-ranked model, H={horizon} ({best_model})")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_dir / f"{safe_name(system)}_H{horizon}_best_fan.png", dpi=180)
            plt.close(fig)


def write_methods(output_dir: Path) -> None:
    text = """# Nodal probabilistic forecasting benchmark

## Interpretation

The primary model ordering is the out-of-sample forecast rank. It averages ranks
for normalized MAE, normalized CRPS, standardized energy score, calibration error,
aggregate normalized CRPS, and normalized Wasserstein distance.

AIC and BIC are secondary fit diagnostics. Compare them only among rows with the
same criterion_scope. Copula AIC/BIC use semiparametric pseudo-likelihoods and do
not include the nonparametric marginal distributions.

Wasserstein distance is computed after pooling the rolling-origin predictive draws
for each node. It is a marginal distribution diagnostic, not a strictly proper
conditional forecast score. CRPS and the energy score are the principal proper
scores.

The energy score evaluates the complete nodal vector. The variogram score is added
because energy scores may have limited sensitivity to dependence misspecification.
System-aggregate metrics test whether spatial scenarios aggregate correctly.

## Implemented approximations

* gaussian_copula: pairwise normal-score correlations projected to a positive
  definite correlation matrix, with empirical innovation marginals.
* student_t_copula: common degrees of freedom selected by pseudo-likelihood;
  pairwise dependence is assembled into a positive definite matrix.
* gmcm: bivariate Gaussian mixtures on the edges of a maximum-spanning tree.
  This avoids pretending that independently fitted pairs automatically define a
  coherent high-dimensional copula.
* conditional_empirical_copula: beta-kernel conditional marginals based on the
  observed one-week lag; a Schaake shuffle restores empirical space-time ranks.
* spatiotemporal_gp: PCA/LMC low-rank multi-output GP for continuous nodal loads.
  It is not the full heterogeneous-likelihood HetMOGP because all outputs here
  have the same continuous data type.

## Output files

* summary_metrics.csv: model-level out-of-sample metrics and ranks.
* model_comparison.csv: forecast metrics joined to likelihood/AIC/BIC diagnostics.
* overall_model_ranking.csv: cross-system and cross-horizon forecast ranking.
* nodal_metrics.csv: every node, model, system and horizon.
* fit_information.csv: likelihood, AIC/BIC and criterion scope.
* forecast_examples.csv: aggregate fan-chart data for the first evaluation origin.
* data_summary.csv: source, shape and chronological split.
* errors.csv: any model/system failures without discarding successful results.
* run_manifest.json: complete run configuration and software versions.
"""
    (output_dir / "METHODS.md").write_text(text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=["IEEE24", "IEEE118"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 24, 168])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--profile", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--n-scenarios", type=int, default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--validation-end-fraction", type=float, default=0.80)
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--gmcm-max-components", type=int, default=None)
    parser.add_argument("--gp-rank", type=int, default=None)
    parser.add_argument("--gp-max-train", type=int, default=None)
    parser.add_argument("--conditional-bandwidth", type=float, default=0.10)
    parser.add_argument("--max-variogram-pairs", type=int, default=250)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def config_from_arguments(args: argparse.Namespace) -> Config:
    profiles = {
        "quick": dict(n_scenarios=50, max_origins=3, gmcm_max_components=2, gp_rank=4, gp_max_train=180),
        "standard": dict(n_scenarios=200, max_origins=20, gmcm_max_components=3, gp_rank=8, gp_max_train=400),
        "full": dict(n_scenarios=500, max_origins=50, gmcm_max_components=4, gp_rank=12, gp_max_train=700),
    }
    selected = profiles[args.profile]
    return Config(
        systems=tuple(args.systems),
        horizons=tuple(sorted(set(args.horizons))),
        models=tuple(args.models),
        n_scenarios=args.n_scenarios or selected["n_scenarios"],
        max_origins=args.max_origins if args.max_origins is not None else selected["max_origins"],
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_end_fraction=args.validation_end_fraction,
        ridge_alphas=tuple(args.ridge_alphas),
        gmcm_max_components=args.gmcm_max_components or selected["gmcm_max_components"],
        gp_rank=args.gp_rank or selected["gp_rank"],
        gp_max_train=args.gp_max_train or selected["gp_max_train"],
        conditional_bandwidth=args.conditional_bandwidth,
        max_variogram_pairs=args.max_variogram_pairs,
        save_models=args.save_models,
        make_plots=not args.no_plots,
    )


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = config_from_arguments(args)
    if any(h < 1 or h > 168 for h in config.horizons):
        raise ValueError("This benchmark supports horizons from 1 through 168 hours.")
    if not (0.3 < config.train_fraction < config.validation_end_fraction < 0.95):
        raise ValueError("Require 0.3 < train_fraction < validation_end_fraction < 0.95.")
    if config.n_scenarios < 20:
        raise ValueError("Use at least 20 scenarios for probabilistic metrics.")
    if not (0.0 < config.conditional_bandwidth <= 1.0):
        raise ValueError("conditional_bandwidth must lie in (0, 1].")

    project_root = detect_project_root()
    data_root = (args.data_root or (project_root / "data" / "powersystems")).resolve()
    output_dir = (args.output_dir or (project_root / "outputs" / "nodal_forecasting")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Data root: %s", data_root)
    LOGGER.info("Output directory: %s", output_dir)

    summary_rows: List[Dict[str, Any]] = []
    nodal_frames: List[pd.DataFrame] = []
    fit_rows: List[Dict[str, Any]] = []
    example_frames: List[pd.DataFrame] = []
    data_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for system in config.systems:
        try:
            dataset = load_dataset(data_root, system)
            y = dataset.values
            n_time = y.shape[0]
            maximum_horizon = max(config.horizons)
            train_end = int(config.train_fraction * n_time)
            fit_end = int(config.validation_end_fraction * n_time)
            fit_end = min(fit_end, n_time - maximum_horizon)
            if train_end <= max(config.lags) + 48 or fit_end <= train_end + 48:
                raise ValueError(
                    f"Insufficient observations for split: T={n_time}, train_end={train_end}, fit_end={fit_end}"
                )
            calendar = calendar_features(n_time, dataset.timestamps)
            hours, weekend = time_components(n_time, dataset.timestamps)
            node_scale = np.std(y[:fit_end], axis=0)
            positive_scale = node_scale[node_scale > EPS]
            scale_floor = max(float(np.median(positive_scale)) * 1.0e-3, 1.0e-8)
            node_scale = np.maximum(node_scale, scale_floor)
            pairs = select_variogram_pairs(y[:fit_end], config.max_variogram_pairs)

            data_rows.append(
                {
                    "system": system,
                    "source_path": str(dataset.source_path),
                    "n_hours": n_time,
                    "n_nodes": y.shape[1],
                    "train_end": train_end,
                    "fit_end_after_validation": fit_end,
                    "test_hours": n_time - fit_end,
                    "first_timestamp": str(dataset.timestamps[0]) if dataset.timestamps is not None else None,
                    "last_timestamp": str(dataset.timestamps[-1]) if dataset.timestamps is not None else None,
                }
            )

            LOGGER.info("%s: fitting requested forecasters", system)
            forecasters = build_forecasters(
                config, y, train_end, fit_end, calendar, hours, weekend, system
            )
            for model_name, forecaster in forecasters.items():
                info = dict(forecaster.fit_info)
                info["system"] = system
                fit_rows.append(info)
                if config.save_models:
                    try:
                        import joblib

                        model_dir = output_dir / "models" / system
                        model_dir.mkdir(parents=True, exist_ok=True)
                        joblib.dump(forecaster, model_dir / f"{model_name}.joblib", compress=3)
                    except Exception as exc:
                        error_rows.append(
                            {
                                "system": system,
                                "model": model_name,
                                "horizon": None,
                                "stage": "save_model",
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )

            for horizon in config.horizons:
                origins = make_origins(fit_end, n_time, horizon, config.max_origins)
                if origins.size == 0:
                    LOGGER.warning("%s H=%d: no valid test origins", system, horizon)
                    continue
                for model_name, forecaster in forecasters.items():
                    LOGGER.info(
                        "%s | H=%d | %s | origins=%d | scenarios=%d",
                        system,
                        horizon,
                        model_name,
                        origins.size,
                        config.n_scenarios,
                    )
                    try:
                        summary, nodal, examples = evaluate_forecaster(
                            forecaster,
                            system,
                            y,
                            dataset.node_names,
                            dataset.timestamps,
                            origins,
                            horizon,
                            config.n_scenarios,
                            node_scale,
                            pairs,
                            stable_seed(config.seed, system, model_name, horizon),
                        )
                        summary_rows.append(summary)
                        nodal_frames.append(nodal)
                        example_frames.append(examples)
                    except Exception as exc:
                        LOGGER.exception("Evaluation failed: %s H=%d %s", system, horizon, model_name)
                        error_rows.append(
                            {
                                "system": system,
                                "model": model_name,
                                "horizon": horizon,
                                "stage": "evaluate",
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
        except Exception as exc:
            LOGGER.exception("System failed: %s", system)
            error_rows.append(
                {
                    "system": system,
                    "model": None,
                    "horizon": None,
                    "stage": "load_or_fit_system",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    summary = pd.DataFrame(summary_rows)
    nodal = pd.concat(nodal_frames, ignore_index=True) if nodal_frames else pd.DataFrame()
    fit_info = pd.DataFrame(fit_rows)
    examples = pd.concat(example_frames, ignore_index=True) if example_frames else pd.DataFrame()
    data_summary = pd.DataFrame(data_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=["system", "model", "horizon", "stage", "error", "traceback"],
    )

    if not summary.empty:
        summary = add_skill_scores_and_ranks(summary)
        summary = summary.sort_values(["system", "horizon", "forecast_rank", "model"])
    if not fit_info.empty:
        fit_info = rank_fit_information(fit_info)
        fit_info = fit_info.sort_values(["system", "criterion_scope", "bic"], na_position="last")

    comparison = build_model_comparison(summary, fit_info)
    if not comparison.empty:
        comparison = comparison.sort_values(["system", "horizon", "forecast_rank", "model"])
    overall_ranking = build_overall_ranking(summary, fit_info)

    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    overall_ranking.to_csv(output_dir / "overall_model_ranking.csv", index=False)
    nodal.to_csv(output_dir / "nodal_metrics.csv", index=False)
    fit_info.to_csv(output_dir / "fit_information.csv", index=False)
    examples.to_csv(output_dir / "forecast_examples.csv", index=False)
    data_summary.to_csv(output_dir / "data_summary.csv", index=False)
    errors.to_csv(output_dir / "errors.csv", index=False)

    manifest = {
        "configuration": asdict(config),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
        },
        "primary_ranking_metrics": [
            "nmae",
            "ncrps",
            "energy_score_standardized",
            "mace",
            "aggregate_ncrps",
            "normalized_wasserstein",
        ],
        "references": [
            "https://arxiv.org/abs/2003.03835",
            "https://proceedings.neurips.cc/paper/2018/hash/165a59f7cf3b5c4396ba65953d679f17-Abstract.html",
            "https://doi.org/10.1016/j.segan.2025.101708",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    write_methods(output_dir)
    if config.make_plots and not summary.empty and not examples.empty:
        make_figures(summary, examples, output_dir)

    if summary.empty:
        LOGGER.error("No successful benchmark result. Inspect %s", output_dir / "errors.csv")
        return 2
    LOGGER.info("Completed. Main ranking: %s", output_dir / "summary_metrics.csv")
    if not errors.empty:
        LOGGER.warning("%d failures were recorded in %s", len(errors), output_dir / "errors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
