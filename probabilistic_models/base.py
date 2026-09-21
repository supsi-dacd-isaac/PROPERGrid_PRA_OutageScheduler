"""Common interfaces for probabilistic scenario models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class ModelNotFittedError(RuntimeError):
    """Raised when prediction or sampling is requested before fitting."""


def as_2d_float(values: ArrayLike, *, name: str = "values") -> FloatArray:
    """Return finite observations in ``(time, variables)`` orientation."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; received shape {array.shape}.")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least 2 rows and 1 column.")
    if not np.all(np.isfinite(array)):
        bad = int(np.size(array) - np.isfinite(array).sum())
        raise ValueError(f"{name} contains {bad} non-finite entries.")
    return array


def as_rng(random_state: int | np.random.Generator | None = None) -> np.random.Generator:
    """Normalize seed-like inputs without touching NumPy's global RNG state."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class NodalScenarioModel(ABC):
    """Interface for unconditional multivariate nodal scenario generators.

    The research demos fit a distribution to rows of a nodal matrix and draw
    coherent system-wide scenarios. Conditional forecasters can implement the
    same interface after first converting observations to forecast residuals.
    """

    name = "abstract"

    def __init__(self, *, nonnegative: bool = True, constant_tolerance: float = 1e-12):
        self.nonnegative = bool(nonnegative)
        self.constant_tolerance = float(constant_tolerance)
        self.n_features_: int | None = None
        self.active_mask_: NDArray[np.bool_] | None = None
        self.constant_values_: FloatArray | None = None
        self.metadata_: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.n_features_ is not None

    def _prepare_fit(self, values: ArrayLike) -> tuple[FloatArray, FloatArray]:
        data = as_2d_float(values)
        scale = np.std(data, axis=0, ddof=1)
        self.active_mask_ = scale > self.constant_tolerance
        self.constant_values_ = np.mean(data, axis=0)
        self.n_features_ = data.shape[1]
        active = data[:, self.active_mask_]
        self.metadata_ = {
            "n_observations": int(data.shape[0]),
            "n_features": int(data.shape[1]),
            "n_active_features": int(self.active_mask_.sum()),
            "constant_columns": np.flatnonzero(~self.active_mask_).astype(int).tolist(),
        }
        return data, active

    def _restore_columns(self, active_samples: FloatArray, n_samples: int) -> FloatArray:
        if not self.is_fitted or self.active_mask_ is None or self.constant_values_ is None:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        result = np.broadcast_to(self.constant_values_, (n_samples, self.n_features_)).copy()
        result[:, self.active_mask_] = active_samples
        if self.nonnegative:
            np.maximum(result, 0.0, out=result)
        return result

    @abstractmethod
    def fit(self, values: ArrayLike) -> "NodalScenarioModel":
        """Fit the model from a ``(time, nodes)`` matrix."""

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        random_state: int | np.random.Generator | None = None,
    ) -> FloatArray:
        """Draw a ``(n_samples, nodes)`` scenario matrix."""

    def predict(self) -> FloatArray:
        """Return the unconditional mean used as a point prediction."""

        if not self.is_fitted or self.constant_values_ is None:
            raise ModelNotFittedError(f"{self.name} has not been fitted.")
        return self.constant_values_.copy()

    def save(self, filepath: str | Path) -> None:
        """Serialize a fitted model with Python pickle."""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str | Path) -> "NodalScenarioModel":
        """Load a model and check that it follows this interface."""

        with Path(filepath).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, NodalScenarioModel):
            raise TypeError("The serialized object is not a NodalScenarioModel.")
        return model

