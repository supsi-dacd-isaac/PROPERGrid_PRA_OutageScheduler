"""Nodal scenario models with explicit marginal/dependence assumptions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln
from scipy.stats import norm, rankdata, t

from .base import FloatArray, ModelNotFittedError, NodalScenarioModel, as_rng


def _regularize_correlation(correlation: FloatArray, minimum_eigenvalue: float) -> FloatArray:
    correlation = np.asarray(correlation, dtype=float)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, minimum_eigenvalue)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(diagonal, diagonal)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _empirical_ppf(sorted_values: FloatArray, uniforms: FloatArray) -> FloatArray:
    n_observations, n_features = sorted_values.shape
    if uniforms.shape[1] != n_features:
        raise ValueError("Uniform samples and marginal distributions have incompatible dimensions.")
    positions = np.clip(uniforms, 0.0, 1.0) * (n_observations - 1)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, n_observations - 1)
    weight = positions - lower
    columns = np.arange(n_features)[None, :]
    return (1.0 - weight) * sorted_values[lower, columns] + weight * sorted_values[upper, columns]


def _pseudo_observations(values: FloatArray) -> FloatArray:
    ranks = rankdata(values, method="average", axis=0)
    return np.clip(ranks / (values.shape[0] + 1.0), 1e-9, 1.0 - 1e-9)


def _multivariate_t_logpdf(scores: FloatArray, correlation: FloatArray, degrees_of_freedom: float) -> FloatArray:
    dimension = scores.shape[1]
    sign, log_determinant = np.linalg.slogdet(correlation)
    if sign <= 0:
        raise ValueError("Student-t copula correlation matrix is not positive definite.")
    quadratic = np.einsum("ij,jk,ik->i", scores, np.linalg.inv(correlation), scores)
    nu = degrees_of_freedom
    constant = (
        gammaln((nu + dimension) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * log_determinant
        - 0.5 * dimension * np.log(nu * np.pi)
    )
    return constant - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)


def default_model_suite() -> list[NodalScenarioModel]:
    """Return the pre-declared model set used in formal comparisons."""

    return [
        IndependentGaussianModel(),
        IndependentEmpiricalModel(),
        MultivariateGaussianModel(),
        GaussianCopulaModel(),
        StudentTCopulaModel(),
        JointEmpiricalBootstrapModel(),
    ]


class IndependentGaussianModel(NodalScenarioModel):
    """Independent Gaussian marginals; a deliberately simple baseline."""

    name = "independent_gaussian"

    def fit(self, values: ArrayLike) -> "IndependentGaussianModel":
        _, active = self._prepare_fit(values)
        self.mean_ = active.mean(axis=0)
        self.std_ = active.std(axis=0, ddof=1)
        return self

    def sample(self, n_samples: int, random_state=None) -> FloatArray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        rng = as_rng(random_state)
        active = rng.normal(self.mean_, self.std_, size=(int(n_samples), self.mean_.size))
        return self._restore_columns(active, int(n_samples))


class MultivariateGaussianModel(NodalScenarioModel):
    """Regularized multivariate Gaussian model in physical units."""

    name = "multivariate_gaussian"

    def __init__(self, *, covariance_shrinkage: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.covariance_shrinkage = float(covariance_shrinkage)

    def fit(self, values: ArrayLike) -> "MultivariateGaussianModel":
        _, active = self._prepare_fit(values)
        self.mean_ = active.mean(axis=0)
        if active.shape[1] == 1:
            covariance = np.array([[float(np.var(active[:, 0], ddof=1))]])
        else:
            covariance = np.cov(active, rowvar=False, ddof=1)
        diagonal = np.diag(np.diag(covariance))
        self.covariance_ = (1.0 - self.covariance_shrinkage) * covariance + self.covariance_shrinkage * diagonal
        return self

    def sample(self, n_samples: int, random_state=None) -> FloatArray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        rng = as_rng(random_state)
        active = rng.multivariate_normal(
            self.mean_, self.covariance_, size=int(n_samples), check_valid="raise", method="eigh"
        )
        if active.ndim == 1:
            active = active[:, None]
        return self._restore_columns(active, int(n_samples))


class IndependentEmpiricalModel(NodalScenarioModel):
    """Empirical marginals sampled independently; dependence-control baseline."""

    name = "independent_empirical"

    def fit(self, values: ArrayLike) -> "IndependentEmpiricalModel":
        _, active = self._prepare_fit(values)
        self.sorted_values_ = np.sort(active, axis=0)
        return self

    def sample(self, n_samples: int, random_state=None) -> FloatArray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        rng = as_rng(random_state)
        uniforms = rng.random((int(n_samples), self.sorted_values_.shape[1]))
        active = _empirical_ppf(self.sorted_values_, uniforms)
        return self._restore_columns(active, int(n_samples))


class JointEmpiricalBootstrapModel(NodalScenarioModel):
    """Row bootstrap preserving the observed joint distribution exactly."""

    name = "joint_empirical_bootstrap"

    def fit(self, values: ArrayLike) -> "JointEmpiricalBootstrapModel":
        data, active = self._prepare_fit(values)
        self.active_rows_ = active.copy()
        return self

    def sample(self, n_samples: int, random_state=None) -> FloatArray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        rng = as_rng(random_state)
        indices = rng.integers(0, self.active_rows_.shape[0], size=int(n_samples))
        return self._restore_columns(self.active_rows_[indices], int(n_samples))


class GaussianCopulaModel(NodalScenarioModel):
    """Gaussian copula with non-parametric empirical marginal distributions."""

    name = "gaussian_copula_empirical"

    def __init__(self, *, minimum_eigenvalue: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.minimum_eigenvalue = float(minimum_eigenvalue)

    def fit(self, values: ArrayLike) -> "GaussianCopulaModel":
        _, active = self._prepare_fit(values)
        self.sorted_values_ = np.sort(active, axis=0)
        uniforms = _pseudo_observations(active)
        gaussian_scores = norm.ppf(uniforms)
        if active.shape[1] == 1:
            correlation = np.ones((1, 1))
        else:
            correlation = np.corrcoef(gaussian_scores, rowvar=False)
        self.correlation_ = _regularize_correlation(correlation, self.minimum_eigenvalue)
        return self

    def sample(self, n_samples: int, random_state=None) -> FloatArray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        rng = as_rng(random_state)
        scores = rng.multivariate_normal(
            np.zeros(self.correlation_.shape[0]),
            self.correlation_,
            size=int(n_samples),
            check_valid="raise",
            method="eigh",
        )
        if scores.ndim == 1:
            scores = scores[:, None]
        active = _empirical_ppf(self.sorted_values_, norm.cdf(scores))
        return self._restore_columns(active, int(n_samples))


class StudentTCopulaModel(NodalScenarioModel):
    """Student-t copula with empirical marginals and optional df selection."""

    name = "student_t_copula_empirical"

    def __init__(
        self,
        *,
        degrees_of_freedom: float | None = None,
        df_candidates: Iterable[float] = (3.0, 4.0, 6.0, 10.0, 20.0, 50.0),
        minimum_eigenvalue: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.degrees_of_freedom = degrees_of_freedom
        self.df_candidates = tuple(float(value) for value in df_candidates)
        self.minimum_eigenvalue = float(minimum_eigenvalue)

    def _fit_candidate(self, uniforms: FloatArray, degrees_of_freedom: float) -> tuple[FloatArray, float]:
        scores = t.ppf(uniforms, df=degrees_of_freedom)
        if scores.shape[1] == 1:
            correlation = np.ones((1, 1))
        else:
            correlation = np.corrcoef(scores, rowvar=False)
        correlation = _regularize_correlation(correlation, self.minimum_eigenvalue)
        joint = _multivariate_t_logpdf(scores, correlation, degrees_of_freedom)
        marginal = np.sum(t.logpdf(scores, df=degrees_of_freedom), axis=1)
        return correlation, float(np.mean(joint - marginal))

    def fit(self, values: ArrayLike) -> "StudentTCopulaModel":
        _, active = self._prepare_fit(values)
        self.sorted_values_ = np.sort(active, axis=0)
        uniforms = _pseudo_observations(active)
        candidates = (
            (float(self.degrees_of_freedom),)
            if self.degrees_of_freedom is not None
            else self.df_candidates
        )
        fitted = []
        for degrees_of_freedom in candidates:
            if degrees_of_freedom <= 2:
                raise ValueError("Student-t degrees of freedom must exceed 2.")
            correlation, objective = self._fit_candidate(uniforms, degrees_of_freedom)
            fitted.append((objective, degrees_of_freedom, correlation))
        objective, self.degrees_of_freedom_, self.correlation_ = max(fitted, key=lambda item: item[0])
        self.metadata_["selected_degrees_of_freedom"] = self.degrees_of_freedom_
        self.metadata_["mean_copula_log_likelihood"] = objective
        return self

    def sample(self, n_samples: int, random_state=None) -> FloatArray:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        rng = as_rng(random_state)
        normal_draws = rng.multivariate_normal(
            np.zeros(self.correlation_.shape[0]),
            self.correlation_,
            size=int(n_samples),
            check_valid="raise",
            method="eigh",
        )
        if normal_draws.ndim == 1:
            normal_draws = normal_draws[:, None]
        chi_square = rng.chisquare(self.degrees_of_freedom_, size=int(n_samples))
        scores = normal_draws / np.sqrt(chi_square[:, None] / self.degrees_of_freedom_)
        active = _empirical_ppf(self.sorted_values_, t.cdf(scores, df=self.degrees_of_freedom_))
        return self._restore_columns(active, int(n_samples))

