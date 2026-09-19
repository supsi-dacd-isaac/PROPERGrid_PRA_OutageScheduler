"""Compatibility access to the revised Poisson/Markov contingency models."""

from .contingencies import (  # noqa: F401
    ComponentSpec,
    ComponentState,
    ComponentType,
    FailureRateTable,
    PoissonFailureModel,
    ThreeStateMarkovModel,
    fit_poisson_rate,
    transition_table,
)


def markov_contingency_model_normal(*args, **kwargs):
    """Build a CTMC evaluated with the H2/normal rate when queried."""

    return ThreeStateMarkovModel(*args, **kwargs)


def markov_contingency_model_weather_induced(*args, **kwargs):
    """Build the same CTMC; pass H1/adverse to its evaluation methods."""

    return ThreeStateMarkovModel(*args, **kwargs)

