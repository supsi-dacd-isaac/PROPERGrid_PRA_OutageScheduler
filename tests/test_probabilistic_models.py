from __future__ import annotations

import numpy as np
import pytest

from probabilistic_models.demand_sampler import (
    simple_demand_sampler,
    transform_to_fat_tailed,
)


@pytest.mark.unit
def test_fat_tail_transform_preserves_sign_and_increases_large_magnitudes() -> None:
    values = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0])
    transformed = transform_to_fat_tailed(values, power=2.0)
    np.testing.assert_allclose(transformed, [-4.0, -0.25, 0.0, 0.25, 4.0])


@pytest.mark.unit
def test_simple_sampler_is_reproducible_for_vector_demand() -> None:
    mean = np.asarray([10.0, 20.0, 30.0])
    first = simple_demand_sampler(mean, n_samples=5, noise_level=0.1, random_seed=7)
    second = simple_demand_sampler(mean, n_samples=5, noise_level=0.1, random_seed=7)
    assert len(first) == 5
    assert all(np.asarray(sample).shape == (3,) for sample in first)
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))


@pytest.mark.unit
def test_zero_noise_returns_the_mean() -> None:
    samples = simple_demand_sampler(42.0, n_samples=4, noise_level=0.0)
    assert samples == pytest.approx([42.0] * 4)


@pytest.mark.unit
def test_fat_tailed_path_is_reproducible() -> None:
    first = simple_demand_sampler(100.0, 6, fat_tailed=True, random_seed=11)
    second = simple_demand_sampler(100.0, 6, fat_tailed=True, random_seed=11)
    assert first == pytest.approx(second)
