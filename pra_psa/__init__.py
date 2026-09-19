"""PROPER tools for probabilistic risk assessment of power grids.

The initializer stays dependency-light. Legacy modules remain importable from
their original paths but are not imported eagerly because they require optional
solver and repository-specific dependencies.
"""

from pra_psa.time_series_pra import (
    GaussianRelativeLoadSampler,
    PRAConfig,
    TimeSeriesPRAResult,
    UniformProbabilityModel,
    WeatherConditionedPoissonModel,
    run_time_series_pra,
)

__version__ = "0.2.0"
__author__ = "PROPERGrid Team"

__all__ = [
    "GaussianRelativeLoadSampler",
    "PRAConfig",
    "TimeSeriesPRAResult",
    "UniformProbabilityModel",
    "WeatherConditionedPoissonModel",
    "run_time_series_pra",
]
