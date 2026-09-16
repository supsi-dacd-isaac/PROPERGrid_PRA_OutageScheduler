"""Deprecated import shim for the original misspelled module name.

New code should import from :mod:`probabilistic_models.nodal` and use the
``fit(values)`` / ``sample(n_samples)`` contract.
"""

from __future__ import annotations

import warnings

import numpy as np

from .nodal import GaussianCopulaModel


class Probability_model_nodal_load(GaussianCopulaModel):
    """Compatibility wrapper around the revised Gaussian-copula model."""

    def __init__(self, net=None, *args, **kwargs):
        ignored = {key: kwargs.pop(key) for key in list(kwargs) if key in {
            "window_size_past_x", "horizon_prediction_steps", "model_class", "name"
        }}
        if args or ignored:
            warnings.warn(
                "Legacy forecasting arguments are ignored. Fit an explicit conditional mean model first "
                "and pass its residual matrix to GaussianCopulaModel when forecasting is required.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)
        self.net = net

    def fit_marginals(self, values):
        return self.fit(values)


class Demand_sampler:
    """Legacy network wrapper with an explicit fitted/fallback distinction."""

    def __init__(self, net, model=None, *, random_state: int = 42):
        self.net = net
        self.prob_model = model or GaussianCopulaModel()
        self.random_state = int(random_state)
        self.is_fitted = False
        self.mean_p = np.asarray(net.load["p_mw"], dtype=float)
        self.mean_q = np.asarray(net.load.get("q_mvar", np.zeros_like(self.mean_p)), dtype=float)

    def fit(self, data):
        self.prob_model.fit(data)
        self.is_fitted = True
        return self.prob_model

    def generate_samples(self, num_samples: int = 1000):
        if self.is_fitted:
            p = self.prob_model.sample(num_samples, random_state=self.random_state)
        else:
            warnings.warn(
                "PLACEHOLDER: no fitted nodal data; using independent 10% Gaussian demand errors.",
                RuntimeWarning,
                stacklevel=2,
            )
            rng = np.random.default_rng(self.random_state)
            p = rng.normal(self.mean_p, 0.1 * np.abs(self.mean_p), size=(num_samples, self.mean_p.size))
            p = np.maximum(p, 0.0)
        ratio = np.divide(self.mean_q, self.mean_p, out=np.zeros_like(self.mean_q), where=self.mean_p != 0)
        q = p * ratio
        return p, q


ProbabilisticModel = GaussianCopulaModel

