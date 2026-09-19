"""
Subset Simulation sampler for rare-event probability estimation.

The sampler estimates

    Pf = P[g(U) >= failure_threshold],     U ~ N(0, I)

where g is a scalar limit-state / severity function. Larger g means more severe
operation; failure is normally g >= 0.

Typical use in power-system PRA:
    - U contains standard-normal uncertainty variables for load or renewable
      uncertainty.
    - limit_state(U) maps U to a sampled operating state, runs PF/OPF/PF under
      the required contingency, and returns a scalar severity margin such as
      max(line_loading_percent - 100).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import copy
import math
import numpy as np
from pathlib import Path

ArrayLike = np.ndarray
LimitState = Callable[[ArrayLike], float]


@dataclass(frozen=True)
class SubsetLevel:
    """Diagnostic data for one subset-simulation level."""

    level: int
    threshold: float
    conditional_probability: float
    n_samples: int
    n_seeds: int
    n_failures: int
    acceptance_rate: float
    mean_g: float
    max_g: float


@dataclass
class SubsetSimulationResult:
    """Subset Simulation output."""

    probability: float
    beta: float
    failure_threshold: float
    levels: list[SubsetLevel]
    samples_u: np.ndarray
    samples_g: np.ndarray
    failure_mask: np.ndarray
    n_model_evaluations: int
    coefficient_of_variation_approx: float

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def log_probability(self) -> float:
        if self.probability <= 0.0:
            return -np.inf
        return float(np.log(self.probability))

    def summary(self) -> dict[str, float | int]:
        return {
            "probability": float(self.probability),
            "log_probability": float(self.log_probability),
            "beta": float(self.beta),
            "failure_threshold": float(self.failure_threshold),
            "n_levels": int(self.n_levels),
            "n_model_evaluations": int(self.n_model_evaluations),
            "coefficient_of_variation_approx": float(
                self.coefficient_of_variation_approx
            ),
            "last_threshold": float(self.levels[-1].threshold)
            if self.levels
            else float("nan"),
            "last_acceptance_rate": float(self.levels[-1].acceptance_rate)
            if self.levels
            else float("nan"),
        }


class SubsetSimulationSampler:
    """
    Generic Au-Beck Subset Simulation sampler in standard-normal space.

    Parameters
    ----------
    limit_state:
        Callable receiving one vector ``u`` with shape ``(dimension,)`` and
        returning scalar severity ``g(u)``. Failure is ``g >= failure_threshold``.
    dimension:
        Number of standard-normal uncertain variables.
    n_samples:
        Number of samples per subset level. Use at least 500 for real studies;
        1000--3000 is more stable for small probabilities.
    p0:
        Target conditional probability per level, usually 0.1 or 0.2.
    proposal_std:
        Standard deviation of the Gaussian random-walk proposal in U-space.
    failure_threshold:
        Failure threshold. With the power-grid convention ``g = loading - 100``,
        this is zero.
    max_levels:
        Maximum number of subset levels.
    seed:
        Random seed.
    """

    def __init__(
        self,
        limit_state: LimitState,
        dimension: int,
        *,
        n_samples: int = 1000,
        p0: float = 0.1,
        proposal_std: float | Sequence[float] = 0.8,
        failure_threshold: float = 0.0,
        max_levels: int = 20,
        seed: int | None = None,
        failure_value_on_exception: float | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        if n_samples < 10:
            raise ValueError("n_samples must be at least 10.")
        if not 0.0 < p0 < 1.0:
            raise ValueError("p0 must lie in (0, 1).")
        if max_levels <= 0:
            raise ValueError("max_levels must be positive.")

        self.limit_state = limit_state
        self.dimension = int(dimension)
        self.n_samples = int(n_samples)
        self.p0 = float(p0)
        self.failure_threshold = float(failure_threshold)
        self.max_levels = int(max_levels)
        self.rng = np.random.default_rng(seed)
        self.failure_value_on_exception = failure_value_on_exception

        proposal_std_array = np.asarray(proposal_std, dtype=float)
        if proposal_std_array.ndim == 0:
            proposal_std_array = np.full(self.dimension, float(proposal_std_array))
        if proposal_std_array.shape != (self.dimension,):
            raise ValueError(
                "proposal_std must be scalar or have shape (dimension,)."
            )
        if np.any(proposal_std_array <= 0.0):
            raise ValueError("proposal_std entries must be positive.")
        self.proposal_std = proposal_std_array

        self.n_seeds = max(1, int(round(self.n_samples * self.p0)))
        self.chain_length = int(math.ceil(self.n_samples / self.n_seeds))

    def run(self) -> SubsetSimulationResult:
        """Run Subset Simulation and return probability estimate plus diagnostics."""

        samples_u = self.rng.standard_normal((self.n_samples, self.dimension))
        samples_g = self._evaluate_many(samples_u)
        n_eval = self.n_samples

        levels: list[SubsetLevel] = []
        probability_multiplier = 1.0

        for level in range(self.max_levels + 1):
            failure_mask = samples_g >= self.failure_threshold
            n_failures = int(np.sum(failure_mask))

            seed_indices = self._select_seed_indices(samples_g)
            threshold = float(samples_g[seed_indices].min())
            conditional_probability = float(len(seed_indices) / self.n_samples)

            if threshold >= self.failure_threshold or level == self.max_levels:
                final_conditional_probability = float(n_failures / self.n_samples)
                pf = probability_multiplier * final_conditional_probability

                levels.append(
                    SubsetLevel(
                        level=level,
                        threshold=self.failure_threshold,
                        conditional_probability=final_conditional_probability,
                        n_samples=self.n_samples,
                        n_seeds=len(seed_indices),
                        n_failures=n_failures,
                        acceptance_rate=float("nan"),
                        mean_g=float(np.nanmean(samples_g)),
                        max_g=float(np.nanmax(samples_g)),
                    )
                )

                cov = self._approx_cov(levels, pf)
                return SubsetSimulationResult(
                    probability=float(pf),
                    beta=self.p0,
                    failure_threshold=self.failure_threshold,
                    levels=levels,
                    samples_u=samples_u,
                    samples_g=samples_g,
                    failure_mask=failure_mask,
                    n_model_evaluations=n_eval,
                    coefficient_of_variation_approx=cov,
                )

            levels.append(
                SubsetLevel(
                    level=level,
                    threshold=threshold,
                    conditional_probability=conditional_probability,
                    n_samples=self.n_samples,
                    n_seeds=len(seed_indices),
                    n_failures=n_failures,
                    acceptance_rate=float("nan"),
                    mean_g=float(np.nanmean(samples_g)),
                    max_g=float(np.nanmax(samples_g)),
                )
            )
            probability_multiplier *= conditional_probability

            seeds_u = samples_u[seed_indices]
            seeds_g = samples_g[seed_indices]
            samples_u, samples_g, accepted, proposed = self._sample_conditional_level(
                seeds_u=seeds_u,
                seeds_g=seeds_g,
                threshold=threshold,
            )
            n_eval += proposed

            previous = levels[-1]
            levels[-1] = SubsetLevel(
                level=previous.level,
                threshold=previous.threshold,
                conditional_probability=previous.conditional_probability,
                n_samples=previous.n_samples,
                n_seeds=previous.n_seeds,
                n_failures=previous.n_failures,
                acceptance_rate=float(accepted / proposed) if proposed else 0.0,
                mean_g=previous.mean_g,
                max_g=previous.max_g,
            )

        raise RuntimeError("Subset Simulation terminated unexpectedly.")

    def _select_seed_indices(self, values: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            raise RuntimeError("All limit-state evaluations are non-finite.")

        clean = np.where(finite_mask, values, -np.inf)
        order = np.argsort(clean)
        return order[-self.n_seeds:]

    def _sample_conditional_level(
        self,
        *,
        seeds_u: np.ndarray,
        seeds_g: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate samples conditional on g(U) >= threshold."""

        out_u: list[np.ndarray] = []
        out_g: list[float] = []
        accepted = 0
        proposed = 0

        for seed_u, seed_g in zip(seeds_u, seeds_g):
            current_u = seed_u.copy()
            current_g = float(seed_g)

            out_u.append(current_u.copy())
            out_g.append(current_g)

            for _ in range(self.chain_length - 1):
                proposal_u = current_u + self.proposal_std * self.rng.standard_normal(
                    self.dimension
                )
                proposal_g = self._evaluate_one(proposal_u)
                proposed += 1

                if proposal_g >= threshold:
                    log_alpha = -0.5 * (
                        np.dot(proposal_u, proposal_u)
                        - np.dot(current_u, current_u)
                    )
                    if np.log(self.rng.uniform()) <= min(0.0, log_alpha):
                        current_u = proposal_u
                        current_g = float(proposal_g)
                        accepted += 1

                out_u.append(current_u.copy())
                out_g.append(current_g)

                if len(out_u) >= self.n_samples:
                    break

            if len(out_u) >= self.n_samples:
                break

        return (
            np.asarray(out_u[: self.n_samples], dtype=float),
            np.asarray(out_g[: self.n_samples], dtype=float),
            accepted,
            proposed,
        )

    def _evaluate_one(self, u: np.ndarray) -> float:
        try:
            value = float(self.limit_state(np.asarray(u, dtype=float)))
        except Exception:
            if self.failure_value_on_exception is None:
                raise
            value = float(self.failure_value_on_exception)

        if not np.isfinite(value):
            if self.failure_value_on_exception is None:
                return -np.inf
            return float(self.failure_value_on_exception)
        return value

    def _evaluate_many(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray([self._evaluate_one(u) for u in samples], dtype=float)

    @staticmethod
    def _approx_cov(levels: Sequence[SubsetLevel], pf: float) -> float:
        """
        Crude independence-based coefficient of variation estimate.

        This ignores Markov-chain correlation and should be treated as a quick
        diagnostic, not as a rigorous uncertainty bound.
        """

        if pf <= 0.0 or not levels:
            return float("inf")

        variance_sum = 0.0
        for level in levels:
            p = level.conditional_probability
            n = level.n_samples
            if p <= 0.0 or n <= 0:
                return float("inf")
            variance_sum += (1.0 - p) / (n * p)
        return float(math.sqrt(max(variance_sum, 0.0)))


def SubsetSym(
    limit_state: LimitState,
    dimension: int,
    *,
    n_samples: int = 1000,
    p0: float = 0.1,
    proposal_std: float | Sequence[float] = 0.8,
    failure_threshold: float = 0.0,
    max_levels: int = 20,
    seed: int | None = 42,
    failure_value_on_exception: float | None = None,
) -> SubsetSimulationResult:
    """Backward-compatible functional wrapper."""

    sampler = SubsetSimulationSampler(
        limit_state=limit_state,
        dimension=dimension,
        n_samples=n_samples,
        p0=p0,
        proposal_std=proposal_std,
        failure_threshold=failure_threshold,
        max_levels=max_levels,
        seed=seed,
        failure_value_on_exception=failure_value_on_exception,
    )
    return sampler.run()


# ---------------------------------------------------------------------------
# Optional pandapower helper
# ---------------------------------------------------------------------------
def lognormal_load_multiplier(u: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    """Convert standard-normal variables to mean-one lognormal multipliers."""

    sigma_array = np.asarray(sigma, dtype=float)
    return np.exp(-0.5 * sigma_array**2 + sigma_array * np.asarray(u, dtype=float))


def make_pandapower_loading_limit_state(
    base_network,
    *,
    sigma: float | np.ndarray = 0.05,
    contingency: dict | None = None,
    loading_threshold_percent: float = 100.0,
    pf_solver=None,
    distributed_slack: bool = False,
    failure_on_pf_error: bool = True,
) -> tuple[LimitState, int]:
    """
    Build a simple pandapower limit-state function.

    The uncertain variables are one standard-normal value per load. The function
    maps them to mean-one lognormal load multipliers, runs a power flow, and
    returns

        g(u) = max_line_loading_percent - loading_threshold_percent.

    Failure is therefore g(u) >= 0.
    """

    import pandapower as pp

    solver = pp.rundcpp if pf_solver is None else pf_solver
    nominal_p = base_network.load["p_mw"].to_numpy(dtype=float).copy()
    nominal_q = (
        base_network.load["q_mvar"].to_numpy(dtype=float).copy()
        if "q_mvar" in base_network.load.columns
        else None
    )
    dimension = len(nominal_p)

    def apply_contingency(net, event: dict | None) -> None:
        if event is None:
            return
        element_type = event.get("element_type")
        element_index = event.get("element_index")
        if element_type == "line":
            net.line.at[element_index, "in_service"] = False
        elif element_type == "trafo":
            net.trafo.at[element_index, "in_service"] = False
        elif element_type in {"gen", "generator"}:
            net.gen.at[element_index, "in_service"] = False
        elif element_type == "sgen":
            net.sgen.at[element_index, "in_service"] = False
        else:
            raise ValueError(f"Unsupported contingency element_type: {element_type}")

    def limit_state(u: np.ndarray) -> float:
        net = copy.deepcopy(base_network)
        multipliers = lognormal_load_multiplier(u, sigma=sigma)
        net.load.loc[:, "p_mw"] = nominal_p * multipliers
        if nominal_q is not None:
            net.load.loc[:, "q_mvar"] = nominal_q * multipliers

        apply_contingency(net, contingency)

        try:
            try:
                solver(net, distributed_slack=distributed_slack)
            except TypeError:
                solver(net)
        except Exception:
            if failure_on_pf_error:
                return float("inf")
            raise

        if not getattr(net, "converged", True):
            return float("inf") if failure_on_pf_error else -float("inf")

        loading = np.asarray(net.res_line["loading_percent"], dtype=float)
        if loading.size == 0 or not np.isfinite(loading).any():
            return float("inf") if failure_on_pf_error else -float("inf")

        return float(np.nanmax(loading) - loading_threshold_percent)

    return limit_state, dimension


@dataclass
class MonteCarloResult:
    """Crude Monte Carlo output for comparison with Subset Simulation."""

    probability: float
    n_samples: int
    n_failures: int
    standard_error: float
    coefficient_of_variation: float
    checkpoints: np.ndarray
    convergence: np.ndarray

    def summary(self) -> dict[str, float | int]:
        return {
            "probability": float(self.probability),
            "n_samples": int(self.n_samples),
            "n_failures": int(self.n_failures),
            "standard_error": float(self.standard_error),
            "coefficient_of_variation": float(self.coefficient_of_variation),
        }


def crude_monte_carlo(
    limit_state: LimitState,
    dimension: int,
    *,
    n_samples: int,
    failure_threshold: float = 0.0,
    seed: int | None = None,
    checkpoints: Sequence[int] | None = None,
    failure_value_on_exception: float | None = None,
) -> MonteCarloResult:
    """
    Estimate failure probability by direct Monte Carlo.

    This is mainly used as a baseline. For rare events, it usually needs many
    more model evaluations than Subset Simulation.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)
    u_samples = rng.standard_normal((n_samples, dimension))
    failures = np.zeros(n_samples, dtype=bool)

    for i, u in enumerate(u_samples):
        try:
            g_value = float(limit_state(u))
        except Exception:
            if failure_value_on_exception is None:
                raise
            g_value = float(failure_value_on_exception)

        failures[i] = np.isfinite(g_value) and (g_value >= failure_threshold)
        if not np.isfinite(g_value) and failure_value_on_exception == float("inf"):
            failures[i] = True

    n_failures = int(failures.sum())
    pf = float(n_failures / n_samples)
    se = float(np.sqrt(max(pf * (1.0 - pf), 0.0) / n_samples))
    cov = float(se / pf) if pf > 0.0 else float("inf")

    if checkpoints is None:
        raw = np.unique(
            np.round(np.logspace(1, np.log10(n_samples), 30)).astype(int)
        )
        checkpoints_array = raw[(raw >= 1) & (raw <= n_samples)]
        if checkpoints_array[-1] != n_samples:
            checkpoints_array = np.r_[checkpoints_array, n_samples]
    else:
        checkpoints_array = np.asarray(checkpoints, dtype=int)
        checkpoints_array = checkpoints_array[
            (checkpoints_array >= 1) & (checkpoints_array <= n_samples)
        ]
        if checkpoints_array.size == 0:
            raise ValueError("checkpoints contains no valid sample counts.")

    cumulative_failures = np.cumsum(failures)
    convergence = np.asarray(
        [cumulative_failures[n - 1] / n for n in checkpoints_array],
        dtype=float,
    )

    return MonteCarloResult(
        probability=pf,
        n_samples=int(n_samples),
        n_failures=n_failures,
        standard_error=se,
        coefficient_of_variation=cov,
        checkpoints=checkpoints_array,
        convergence=convergence,
    )


def normal_sum_tail_probability(
    *,
    threshold: float,
    dimension: int,
) -> float:
    """
    Analytical reference for P[sum_i U_i >= threshold], U_i ~ N(0, 1).

    Used only for the demonstration problem in ``__main__``.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    z = threshold / math.sqrt(dimension)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def compare_subset_simulation_with_mc(
    limit_state: LimitState,
    dimension: int,
    *,
    subset_n_samples: int = 1000,
    subset_p0: float = 0.1,
    subset_proposal_std: float = 0.8,
    mc_n_samples: int = 100_000,
    failure_threshold: float = 0.0,
    seed: int = 7,
    exact_probability: float | None = None,
    figure_path: str | Path | None = None,
) -> tuple[SubsetSimulationResult, MonteCarloResult]:
    """
    Run Subset Simulation and crude Monte Carlo on the same limit-state function.

    The optional plot shows direct-MC convergence versus the final Subset
    Simulation estimate at its number of model evaluations.
    """

    subset_result = SubsetSym(
        limit_state,
        dimension=dimension,
        n_samples=subset_n_samples,
        p0=subset_p0,
        proposal_std=subset_proposal_std,
        failure_threshold=failure_threshold,
        seed=seed,
    )

    mc_result = crude_monte_carlo(
        limit_state,
        dimension=dimension,
        n_samples=mc_n_samples,
        failure_threshold=failure_threshold,
        seed=seed + 1,
    )

    print("\nProbability-estimation comparison")
    print("-" * 72)
    if exact_probability is not None:
        print(f"Exact reference probability     : {exact_probability:.8e}")
    print(
        "Subset Simulation estimate      : "
        f"{subset_result.probability:.8e} "
        f"({subset_result.n_model_evaluations} model evaluations)"
    )
    print(
        "Crude Monte Carlo estimate      : "
        f"{mc_result.probability:.8e} "
        f"({mc_result.n_samples} model evaluations, "
        f"{mc_result.n_failures} failures)"
    )
    print(f"MC standard error               : {mc_result.standard_error:.8e}")
    print(f"MC coefficient of variation     : {mc_result.coefficient_of_variation:.3f}")
    print("\nSubset Simulation levels:")
    for level in subset_result.levels:
        print(
            f"  level={level.level:2d}  "
            f"threshold={level.threshold: .5f}  "
            f"cond_p={level.conditional_probability:.4f}  "
            f"failures={level.n_failures:4d}  "
            f"acceptance={level.acceptance_rate:.3f}"
        )

    if figure_path is not None:
        import matplotlib.pyplot as plt

        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            mc_result.checkpoints,
            mc_result.convergence,
            marker="o",
            markersize=3,
            label="Crude Monte Carlo cumulative estimate",
        )
        ax.scatter(
            [subset_result.n_model_evaluations],
            [subset_result.probability],
            marker="s",
            s=60,
            label="Subset Simulation estimate",
        )
        if exact_probability is not None and exact_probability > 0.0:
            ax.axhline(
                exact_probability,
                linestyle="--",
                linewidth=1.2,
                label="Exact reference",
            )

        #ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.set_xlabel("Model evaluations")
        ax.set_ylabel("Estimated failure probability")
        ax.set_title("Subset Simulation versus crude Monte Carlo")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved convergence figure: {figure_path}")

    return subset_result, mc_result


if __name__ == "__main__":
    # Demonstration problem:
    #
    #     g(U) = U1 + U2 - 4,     U ~ N(0, I)
    #
    # Failure is g(U) >= 0, i.e. U1 + U2 >= 4.
    # Since U1 + U2 ~ N(0, 2), the exact probability is available.
    def example_limit_state(u: np.ndarray) -> float:
        return float(u[0] + u[1] - 4.0)

    exact_pf = normal_sum_tail_probability(threshold=4.0, dimension=2)

    subset_result, mc_result = compare_subset_simulation_with_mc(
        example_limit_state,
        dimension=2,
        subset_n_samples=2000,
        subset_p0=0.1,
        subset_proposal_std=0.8,
        mc_n_samples=200_000,
        failure_threshold=0.0,
        seed=7,
        exact_probability=exact_pf,
        figure_path=Path("./outputs/wp2/subset_simulation_demo/convergence_mc_vs_subset.png"),
    )

    print()
    print("Subset summary:")
    print(subset_result.summary())

    print()
    print("Monte Carlo summary:")
    print(mc_result.summary())