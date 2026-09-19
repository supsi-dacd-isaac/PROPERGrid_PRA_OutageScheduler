"""Simple fallback samplers retained for backwards compatibility."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def transform_to_fat_tailed(noise: ArrayLike, power: float = 1.6) -> np.ndarray:
    """Apply a signed power transform to create a demonstrative heavy tail."""

    values = np.asarray(noise, dtype=float)
    if power <= 0:
        raise ValueError("power must be positive.")
    return np.sign(values) * np.abs(values) ** power


def simple_demand_sampler(
    mean_demand_value: ArrayLike,
    n_samples: int,
    noise_level: float = 0.05,
    fat_tailed: bool = False,
    random_seed: int | None = 42,
    *,
    nonnegative: bool = True,
) -> np.ndarray:
    """Sample around one scalar/vector baseline without global RNG side effects.

    This is a placeholder model. Use a fitted nodal model for quantitative PRA.
    """

    mean = np.asarray(mean_demand_value, dtype=float)
    if n_samples < 1 or noise_level < 0 or not np.all(np.isfinite(mean)):
        raise ValueError("Invalid sample count, noise level, or demand baseline.")
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0.0, noise_level * np.maximum(np.abs(mean), 1e-12), size=(n_samples,) + mean.shape)
    if fat_tailed:
        noise = transform_to_fat_tailed(noise)
    samples = mean + noise
    return np.maximum(samples, 0.0) if nonnegative else samples

