"""Poisson failure processes and three-state component Markov models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import gamma

from probabilistic_models.base import as_rng
from probabilistic_models.weather.weather import WeatherRegime


class ComponentType(str, Enum):
    SINGLE_LINE = "single_line"
    MULTIPLE_LINE = "multiple_line"
    BUSBAR_COUPLER = "busbar_coupler"
    TRANSFORMER = "transformer"
    GENERATOR = "generator"
    CUSTOM = "custom"


class ComponentState(IntEnum):
    AVAILABLE = 1
    FORCED_OUTAGE = 2
    PLANNED_OUTAGE = 3


@dataclass(frozen=True)
class RateDefinition:
    rate_per_hour: float
    per_km: bool
    source: str


@dataclass(frozen=True)
class ComponentSpec:
    asset_id: str
    component_type: ComponentType
    length_km: float | None = None
    rate_override_per_hour: dict[str, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class FailureRateTable:
    """Component failure-rate lookup indexed by weather regime and type."""

    def __init__(self, rates: dict[str, dict[ComponentType, RateDefinition]]):
        self.rates = rates

    @classmethod
    def from_supplied_table_3(cls) -> "FailureRateTable":
        """Create the hourly rates shown in the user-supplied Table 3 image.

        Line rates are per km-hour. Busbar-coupler and transformer rates are
        per component-hour. The five-year source data are not included, so the
        values are treated as literature inputs rather than re-fitted estimates.
        """

        source = "Berti et al., supplied Table 3 (five-year historical estimate)"
        h1 = {
            ComponentType.SINGLE_LINE: RateDefinition(6.81e-7, True, source),
            ComponentType.MULTIPLE_LINE: RateDefinition(1.71e-7, True, source),
            ComponentType.BUSBAR_COUPLER: RateDefinition(2.73e-8, False, source),
            ComponentType.TRANSFORMER: RateDefinition(6.27e-6, False, source),
        }
        h2 = {
            ComponentType.SINGLE_LINE: RateDefinition(2.26e-7, True, source),
            ComponentType.MULTIPLE_LINE: RateDefinition(1.77e-8, True, source),
            ComponentType.BUSBAR_COUPLER: RateDefinition(2.73e-8, False, source),
            ComponentType.TRANSFORMER: RateDefinition(6.27e-6, False, source),
        }
        return cls({WeatherRegime.H1_ADVERSE.value: h1, WeatherRegime.H2_NORMAL.value: h2})

    def definition(self, regime: str | WeatherRegime, component_type: ComponentType) -> RateDefinition:
        key = regime.value if isinstance(regime, WeatherRegime) else str(regime)
        try:
            return self.rates[key][component_type]
        except KeyError as error:
            raise KeyError(f"No rate for regime={key!r}, component_type={component_type.value!r}.") from error


@dataclass(frozen=True)
class RateEstimate:
    posterior_mean_per_hour: float
    mle_per_hour: float
    ci95_lower: float
    ci95_upper: float
    event_count: int
    exposure_hours: float
    method: str


def fit_poisson_rate(
    event_count: int,
    exposure_hours: float,
    *,
    prior_shape: float = 0.5,
    prior_rate_hours: float = 0.0,
) -> RateEstimate:
    """Estimate a homogeneous Poisson rate with a Gamma prior.

    The posterior is Gamma(prior_shape + count, prior_rate + exposure),
    where ``rate`` is expressed in hours. Jeffreys' prior is recovered with
    ``prior_shape=0.5`` and ``prior_rate_hours=0``.
    """

    if event_count < 0 or exposure_hours <= 0 or prior_shape <= 0 or prior_rate_hours < 0:
        raise ValueError("Counts/exposures/prior parameters are outside their admissible range.")
    shape = prior_shape + int(event_count)
    rate = prior_rate_hours + float(exposure_hours)
    lower, upper = gamma.ppf([0.025, 0.975], a=shape, scale=1.0 / rate)
    return RateEstimate(
        posterior_mean_per_hour=float(shape / rate),
        mle_per_hour=float(event_count / exposure_hours),
        ci95_lower=float(lower),
        ci95_upper=float(upper),
        event_count=int(event_count),
        exposure_hours=float(exposure_hours),
        method="Gamma-Poisson posterior",
    )


def fit_three_state_transition_rates(
    *,
    available_hours: float,
    forced_outage_hours: float,
    planned_outage_hours: float,
    transitions_12: int,
    transitions_21: int,
    transitions_13: int,
    transitions_31: int,
    prior_shape: float = 0.5,
    prior_rate_hours: float = 0.0,
) -> dict[str, RateEstimate]:
    """Estimate CTMC intensities as transition counts divided by state exposure.

    For transition ``i -> j``, the exposure is the total observation time in
    state ``i``. The same Gamma-Poisson posterior is used for each intensity;
    censoring and interval uncertainty require a richer event-history model.
    """

    return {
        "alpha12_available_to_forced": fit_poisson_rate(
            transitions_12, available_hours, prior_shape=prior_shape, prior_rate_hours=prior_rate_hours
        ),
        "alpha21_forced_to_available": fit_poisson_rate(
            transitions_21, forced_outage_hours, prior_shape=prior_shape, prior_rate_hours=prior_rate_hours
        ),
        "alpha13_available_to_planned": fit_poisson_rate(
            transitions_13, available_hours, prior_shape=prior_shape, prior_rate_hours=prior_rate_hours
        ),
        "alpha31_planned_to_available": fit_poisson_rate(
            transitions_31, planned_outage_hours, prior_shape=prior_shape, prior_rate_hours=prior_rate_hours
        ),
    }


class PoissonFailureModel:
    """Independent component failure processes conditional on a weather regime."""

    def __init__(
        self,
        components: list[ComponentSpec],
        *,
        rate_table: FailureRateTable | None = None,
        default_line_length_km: float | None = None,
    ):
        if not components:
            raise ValueError("At least one component is required.")
        self.components = tuple(components)
        self.by_id = {component.asset_id: component for component in self.components}
        if len(self.by_id) != len(self.components):
            raise ValueError("Component asset_id values must be unique.")
        self.rate_table = rate_table or FailureRateTable.from_supplied_table_3()
        self.default_line_length_km = default_line_length_km
        self.assumption_warnings: list[str] = []

    def rate_per_hour(self, asset_id: str, regime: str | WeatherRegime) -> float:
        component = self.by_id[asset_id]
        regime_key = regime.value if isinstance(regime, WeatherRegime) else str(regime)
        if component.rate_override_per_hour is not None:
            try:
                value = float(component.rate_override_per_hour[regime_key])
            except KeyError as error:
                raise KeyError(f"No rate override for {asset_id!r} in {regime_key!r}.") from error
            if value < 0:
                raise ValueError("Failure rates cannot be negative.")
            return value
        definition = self.rate_table.definition(regime_key, component.component_type)
        if not definition.per_km:
            return definition.rate_per_hour
        length = component.length_km
        if length is None:
            if self.default_line_length_km is None:
                raise ValueError(
                    f"Line length is required for {asset_id!r}; Table 3 rates are per km-hour."
                )
            length = float(self.default_line_length_km)
            message = (
                f"PLACEHOLDER: {asset_id} has no line length; using {length:g} km to scale its failure rate."
            )
            if message not in self.assumption_warnings:
                self.assumption_warnings.append(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        if length < 0:
            raise ValueError("Line length cannot be negative.")
        return definition.rate_per_hour * float(length)

    def rates_per_hour(self, regime: str | WeatherRegime) -> dict[str, float]:
        return {component.asset_id: self.rate_per_hour(component.asset_id, regime) for component in self.components}

    def survival_probability(self, asset_id: str, horizon_hours: float, regime: str | WeatherRegime) -> float:
        if horizon_hours < 0:
            raise ValueError("horizon_hours cannot be negative.")
        return float(np.exp(-self.rate_per_hour(asset_id, regime) * horizon_hours))

    def event_probability(self, asset_id: str, horizon_hours: float, regime: str | WeatherRegime) -> float:
        """Return P(N(horizon)>=1)=1-exp(-lambda*horizon)."""

        return 1.0 - self.survival_probability(asset_id, horizon_hours, regime)

    def contingency_probability(
        self,
        failed_asset_ids: tuple[str, ...] | list[str],
        horizon_hours: float,
        regime: str | WeatherRegime,
        *,
        exact_subset: bool = True,
    ) -> float:
        """Probability of a specified N-k subset under conditional independence."""

        failed = set(failed_asset_ids)
        unknown = failed - self.by_id.keys()
        if unknown:
            raise KeyError(f"Unknown component ids: {sorted(unknown)}")
        probability = 1.0
        for component in self.components:
            survival = self.survival_probability(component.asset_id, horizon_hours, regime)
            if component.asset_id in failed:
                probability *= 1.0 - survival
            elif exact_subset:
                probability *= survival
        return float(probability)

    def normal_probability(self, horizon_hours: float, regime: str | WeatherRegime) -> float:
        total_rate = sum(self.rates_per_hour(regime).values())
        return float(np.exp(-total_rate * horizon_hours))

    def sample_failed_assets(
        self,
        horizon_hours: float,
        regime: str | WeatherRegime,
        *,
        n_samples: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> list[tuple[str, ...]]:
        rng = as_rng(random_state)
        probabilities = np.array(
            [self.event_probability(item.asset_id, horizon_hours, regime) for item in self.components]
        )
        failures = rng.random((int(n_samples), len(self.components))) < probabilities
        identifiers = np.array([item.asset_id for item in self.components], dtype=object)
        return [tuple(identifiers[row].tolist()) for row in failures]


class ThreeStateMarkovModel:
    """Continuous-time Available/Forced/Planned outage component model.

    State ordering and transition directions follow the supplied diagram:
    alpha_12 is forced-failure intensity, alpha_21 forced restoration,
    alpha_13 entry into planned outage, and alpha_31 return from maintenance.
    """

    def __init__(
        self,
        asset_id: str,
        failure_model: PoissonFailureModel,
        *,
        forced_repair_rate_per_hour: float,
        planned_entry_rate_per_hour: float = 0.0,
        planned_return_rate_per_hour: float = 1.0 / 8.0,
        placeholder_parameters: bool = False,
    ):
        rates = (forced_repair_rate_per_hour, planned_entry_rate_per_hour, planned_return_rate_per_hour)
        if any(rate < 0 for rate in rates):
            raise ValueError("Transition rates cannot be negative.")
        if asset_id not in failure_model.by_id:
            raise KeyError(asset_id)
        self.asset_id = asset_id
        self.failure_model = failure_model
        self.alpha21 = float(forced_repair_rate_per_hour)
        self.alpha13 = float(planned_entry_rate_per_hour)
        self.alpha31 = float(planned_return_rate_per_hour)
        self.placeholder_parameters = bool(placeholder_parameters)

    def generator_matrix(self, regime: str | WeatherRegime) -> np.ndarray:
        alpha12 = self.failure_model.rate_per_hour(self.asset_id, regime)
        return np.array(
            [
                [-(alpha12 + self.alpha13), alpha12, self.alpha13],
                [self.alpha21, -self.alpha21, 0.0],
                [self.alpha31, 0.0, -self.alpha31],
            ],
            dtype=float,
        )

    def transition_matrix(self, horizon_hours: float, regime: str | WeatherRegime) -> np.ndarray:
        if horizon_hours < 0:
            raise ValueError("horizon_hours cannot be negative.")
        matrix = expm(self.generator_matrix(regime) * horizon_hours)
        matrix[matrix < 0.0] = 0.0
        return matrix / matrix.sum(axis=1, keepdims=True)

    def stationary_distribution(self, regime: str | WeatherRegime) -> np.ndarray:
        generator = self.generator_matrix(regime)
        augmented = np.vstack([generator.T[:-1], np.ones(3)])
        rhs = np.array([0.0, 0.0, 1.0])
        probability = np.linalg.solve(augmented, rhs)
        return np.maximum(probability, 0.0) / np.maximum(probability, 0.0).sum()

    def transition_table(self, horizon_hours: float, regime: str | WeatherRegime) -> pd.DataFrame:
        labels = [state.name.lower() for state in ComponentState]
        return pd.DataFrame(self.transition_matrix(horizon_hours, regime), index=labels, columns=labels)

    def simulate(
        self,
        n_steps: int,
        step_hours: float,
        regime: str | WeatherRegime | list[str],
        *,
        initial_state: ComponentState = ComponentState.AVAILABLE,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")
        regimes = [regime] * int(n_steps) if isinstance(regime, (str, WeatherRegime)) else list(regime)
        if len(regimes) != int(n_steps):
            raise ValueError("A regime sequence must contain one value per step.")
        rng = as_rng(random_state)
        states = np.empty(int(n_steps) + 1, dtype=np.int8)
        states[0] = int(initial_state) - 1
        for step in range(int(n_steps)):
            transition = self.transition_matrix(step_hours, regimes[step])
            states[step + 1] = rng.choice(3, p=transition[states[step]])
        return states + 1


def transition_table(
    model: ThreeStateMarkovModel,
    horizon_hours: float = 1.0,
    regime: str | WeatherRegime = WeatherRegime.H2_NORMAL,
) -> pd.DataFrame:
    """Compatibility helper replacing the previous empty placeholder."""

    return model.transition_table(horizon_hours, regime)
