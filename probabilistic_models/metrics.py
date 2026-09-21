"""Proper scores and dependence diagnostics for ensemble scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata, wasserstein_distance

from .base import FloatArray, as_2d_float


@dataclass(frozen=True)
class EnsembleEvaluation:
    metrics: dict[str, float]
    energy_score_by_observation: FloatArray
    marginal_crps_by_observation: FloatArray


def _scaling(reference: FloatArray, tolerance: float = 1e-12) -> tuple[FloatArray, FloatArray, np.ndarray]:
    center = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0, ddof=1)
    active = scale > tolerance
    if not np.any(active):
        raise ValueError("All nodal columns are constant; probabilistic scores are undefined.")
    return center[active], scale[active], active


def _half_mean_pairwise_absolute(samples: FloatArray) -> FloatArray:
    """Compute 0.5 E|X-X'| per column in O(m log m)."""

    ordered = np.sort(samples, axis=0)
    m = ordered.shape[0]
    coefficients = 2.0 * np.arange(m) - m + 1.0
    return np.sum(coefficients[:, None] * ordered, axis=0) / (m * m)


def _safe_rank_correlation(values: FloatArray) -> FloatArray:
    ranks = rankdata(values, method="average", axis=0)
    correlation = np.corrcoef(ranks, rowvar=False)
    correlation = np.atleast_2d(correlation)
    return np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_ensemble(
    reference_train: ArrayLike,
    observations: ArrayLike,
    samples: ArrayLike,
    *,
    variogram_power: float = 0.5,
    tail_quantile: float = 0.9,
    chunk_size: int = 256,
) -> EnsembleEvaluation:
    """Evaluate an unconditional multivariate scenario ensemble.

    Energy score, marginal CRPS, and variogram score are computed after
    standardizing active nodes with training statistics. This prevents large
    buses from dominating solely because of their MW scale. Lower is better for
    every error metric; empirical interval coverage is reported separately.
    """

    train = as_2d_float(reference_train, name="reference_train")
    test = as_2d_float(observations, name="observations")
    ensemble = as_2d_float(samples, name="samples")
    if train.shape[1] != test.shape[1] or train.shape[1] != ensemble.shape[1]:
        raise ValueError("Training, test, and sample matrices must have the same node columns.")
    center, scale, active = _scaling(train)
    train_scaled = (train[:, active] - center) / scale
    test_scaled = (test[:, active] - center) / scale
    sample_scaled = (ensemble[:, active] - center) / scale

    m = sample_scaled.shape[0]
    energy_constant = pdist(sample_scaled, metric="euclidean").mean() * (m - 1.0) / (2.0 * m)
    energy_parts = []
    crps_parts = []
    half_pairwise_absolute = _half_mean_pairwise_absolute(sample_scaled)
    for start in range(0, test_scaled.shape[0], chunk_size):
        stop = min(start + chunk_size, test_scaled.shape[0])
        target = test_scaled[start:stop]
        energy_parts.append(cdist(target, sample_scaled, metric="euclidean").mean(axis=1) - energy_constant)
        first_crps_term = np.mean(np.abs(target[:, None, :] - sample_scaled[None, :, :]), axis=1)
        crps_parts.append(np.mean(first_crps_term - half_pairwise_absolute, axis=1))
    energy_by_observation = np.concatenate(energy_parts)
    crps_by_observation = np.concatenate(crps_parts)

    dimension = sample_scaled.shape[1]
    if dimension > 1:
        left, right = np.triu_indices(dimension, 1)
        ensemble_variogram = np.mean(
            np.abs(sample_scaled[:, left] - sample_scaled[:, right]) ** variogram_power,
            axis=0,
        )
        observed_variogram = np.abs(test_scaled[:, left] - test_scaled[:, right]) ** variogram_power
        variogram_by_observation = np.mean((observed_variogram - ensemble_variogram) ** 2, axis=1)

        sample_correlation = _safe_rank_correlation(sample_scaled)
        test_correlation = _safe_rank_correlation(test_scaled)
        off_diagonal = np.triu_indices(dimension, 1)
        spearman_rmse = float(
            np.sqrt(np.mean((sample_correlation[off_diagonal] - test_correlation[off_diagonal]) ** 2))
        )

        # Copula diagnostic: use each distribution's own marginal quantiles so
        # the result measures extremal dependence rather than marginal drift.
        sample_threshold = np.quantile(sample_scaled, tail_quantile, axis=0)
        test_threshold = np.quantile(test_scaled, tail_quantile, axis=0)
        sample_exceedance = sample_scaled > sample_threshold
        test_exceedance = test_scaled > test_threshold
        denominator = max(1.0 - tail_quantile, 1e-9)
        sample_tail = np.mean(sample_exceedance[:, left] & sample_exceedance[:, right], axis=0) / denominator
        test_tail = np.mean(test_exceedance[:, left] & test_exceedance[:, right], axis=0) / denominator
        upper_tail_rmse = float(np.sqrt(np.mean((sample_tail - test_tail) ** 2)))
    else:
        variogram_by_observation = np.zeros(test_scaled.shape[0])
        spearman_rmse = 0.0
        upper_tail_rmse = 0.0

    lower, upper = np.quantile(sample_scaled, [0.05, 0.95], axis=0)
    coverage = float(np.mean((test_scaled >= lower) & (test_scaled <= upper)))
    marginal_wasserstein = float(
        np.mean(
            [
                wasserstein_distance(test_scaled[:, column], sample_scaled[:, column])
                for column in range(dimension)
            ]
        )
    )
    point_rmse = float(np.sqrt(np.mean((test_scaled - sample_scaled.mean(axis=0)) ** 2)))
    metrics = {
        "energy_score": float(np.mean(energy_by_observation)),
        "marginal_crps": float(np.mean(crps_by_observation)),
        "variogram_score_p0.5": float(np.mean(variogram_by_observation)),
        "spearman_correlation_rmse": spearman_rmse,
        "upper_tail_dependence_rmse_q0.9": upper_tail_rmse,
        "marginal_wasserstein": marginal_wasserstein,
        "central_90pct_coverage": coverage,
        "ensemble_mean_rmse": point_rmse,
    }
    return EnsembleEvaluation(metrics, energy_by_observation, crps_by_observation)


def moving_block_bootstrap_mean_difference(
    challenger: ArrayLike,
    reference: ArrayLike,
    *,
    block_length: int = 24,
    n_bootstrap: int = 1000,
    random_state: int = 123,
) -> dict[str, float]:
    """Paired moving-block interval for mean score(challenger)-score(reference)."""

    difference = np.asarray(challenger, dtype=float) - np.asarray(reference, dtype=float)
    if difference.ndim != 1 or difference.size < 2:
        raise ValueError("Paired score vectors must be one-dimensional with at least two values.")
    block_length = int(np.clip(block_length, 1, difference.size))
    starts = np.arange(difference.size - block_length + 1)
    blocks_needed = int(np.ceil(difference.size / block_length))
    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(int(n_bootstrap))
    for replicate in range(int(n_bootstrap)):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        resample = np.concatenate([difference[start : start + block_length] for start in selected])
        bootstrap_means[replicate] = np.mean(resample[: difference.size])
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(difference)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_probability_challenger_worse": float(np.mean(bootstrap_means > 0.0)),
    }
