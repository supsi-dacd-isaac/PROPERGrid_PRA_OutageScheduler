#!/usr/bin/env python3
"""IEEE RTS-24 overload-probability demo.

The script estimates

    P(max branch loading >= threshold)

under a stochastic load model using three estimators:

* crude Monte Carlo (MC);
* Gaussian mean-shift importance sampling (IS); and
* subset simulation (SS) with a component-wise stationary Gaussian kernel.

The same latent-variable model and the same pandapower AC power-flow
evaluation are used by all three methods.  The wind/failure model is not
used here: this is an isolated rare-event-sampling benchmark for the WP2
methodology.  The default load multiplier is intentionally configurable
because the unmodified IEEE RTS-24 case is not necessarily close to its
100-percent loading threshold on every pandapower version.

Example
-------
python demos/subset_simulation_and_importance_sampling.py \
    --samples 1000 --is-samples 1000 --subset-samples 1000 \
    --base-load-scale 1.15 --output outputs/wp2/ieee24_overload.json

The implementation deliberately keeps the sampling code in this demo so it
can be run even while the historical sampler APIs are being consolidated.
Once the public ``pra_psa.sampler`` API is stable, the SS routine can be
replaced by an adapter to the canonical implementation without changing the
network model or the comparison table.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Estimate:
    method: str
    probability: float
    standard_error: float | None
    evaluations: int
    elapsed_seconds: float
    details: dict[str, Any]


class IEEE24OverloadModel:
    """Stochastic IEEE RTS-24 load model evaluated with pandapower."""

    def __init__(
        self,
        net: Any,
        *,
        base_load_scale: float = 1.15,
        sigma: float = 0.12,
        correlation: float = 0.75,
        threshold_percent: float = 100.0,
    ) -> None:
        if not 0.0 <= correlation < 1.0:
            raise ValueError("correlation must satisfy 0 <= correlation < 1")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if base_load_scale <= 0.0:
            raise ValueError("base_load_scale must be positive")

        self.net = net
        self.base_load_scale = float(base_load_scale)
        self.sigma = float(sigma)
        self.correlation = float(correlation)
        self.threshold_percent = float(threshold_percent)
        self.p0 = np.asarray(net.load["p_mw"], dtype=float).copy()
        self.q0 = np.asarray(net.load["q_mvar"], dtype=float).copy()
        if self.p0.size == 0:
            raise ValueError("the IEEE24 network contains no loads")

        # u[0] is a common demand factor; remaining coordinates are
        # load-specific shocks.  This gives a low-dimensional direction for
        # importance sampling while retaining heterogeneous load uncertainty.
        self.dimension = 1 + self.p0.size

    def load_multipliers(self, u: np.ndarray) -> np.ndarray:
        """Return one multiplier per load for one or many latent vectors."""
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            if u.size != self.dimension:
                raise ValueError(f"expected dimension {self.dimension}, got {u.size}")
            common = u[0]
            idiosyncratic = u[1:]
            z = np.sqrt(self.correlation) * common + np.sqrt(
                1.0 - self.correlation
            ) * idiosyncratic
            return np.exp(self.sigma * z - 0.5 * self.sigma**2)
        if u.ndim != 2 or u.shape[1] != self.dimension:
            raise ValueError(f"expected shape (n, {self.dimension}), got {u.shape}")
        common = u[:, [0]]
        idiosyncratic = u[:, 1:]
        z = np.sqrt(self.correlation) * common + np.sqrt(
            1.0 - self.correlation
        ) * idiosyncratic
        return np.exp(self.sigma * z - 0.5 * self.sigma**2)

    @staticmethod
    def _finite_max(values: Any) -> float:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else float("inf")

    def max_loading_percent(self, u: np.ndarray) -> float:
        """Run one AC power flow and return the maximum branch loading."""
        import pandapower as pp  # imported lazily so --help works without it

        out = copy.deepcopy(self.net)
        multiplier = self.load_multipliers(u)
        out.load.loc[:, "p_mw"] = self.p0 * self.base_load_scale * multiplier
        out.load.loc[:, "q_mvar"] = self.q0 * self.base_load_scale * multiplier

        try:
            pp.runpp(
                out,
                init="flat",
                calculate_voltage_angles=False,
                enforce_q_lims=True,
                numba=False,
            )
        except Exception:
            # A non-converged PF is an operational failure and therefore an
            # overload event for this sampling benchmark.
            return float("inf")

        values: list[float] = []
        if hasattr(out, "res_line") and "loading_percent" in out.res_line:
            values.append(self._finite_max(out.res_line.loading_percent))
        if hasattr(out, "res_trafo") and "loading_percent" in out.res_trafo:
            values.append(self._finite_max(out.res_trafo.loading_percent))
        return max(values, default=float("inf"))

    def performance(self, u: np.ndarray) -> float:
        """Limit-state function; the overload event is ``g(u) <= 0``."""
        return self.threshold_percent - self.max_loading_percent(u)

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            return np.asarray([self.performance(u)])
        return np.asarray([self.performance(row) for row in u], dtype=float)


def load_ieee24() -> Any:
    try:
        import pandapower.networks as pn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "This demo requires pandapower. Install the project dependencies "
            "or run `python -m pip install pandapower`."
        ) from exc

    loader = getattr(pn, "case24_ieee_rts", None)
    if loader is None:
        raise RuntimeError("The installed pandapower version has no case24_ieee_rts")
    return loader()


def crude_monte_carlo(
    model: IEEE24OverloadModel, n: int, rng: np.random.Generator
) -> Estimate:
    start = time.perf_counter()
    u = rng.normal(size=(n, model.dimension))
    g = model.evaluate(u)
    indicator = g <= 0.0
    probability = float(np.mean(indicator))
    standard_error = math.sqrt(probability * (1.0 - probability) / n)
    return Estimate(
        method="MC",
        probability=probability,
        standard_error=standard_error,
        evaluations=n,
        elapsed_seconds=time.perf_counter() - start,
        details={"events": int(np.sum(indicator))},
    )


def importance_sampling(
    model: IEEE24OverloadModel,
    n: int,
    shift: float,
    rng: np.random.Generator,
) -> Estimate:
    """Estimate the event under N(mu,I), correcting with exact likelihoods."""
    start = time.perf_counter()
    mu = np.zeros(model.dimension)
    mu[0] = float(shift)
    u = rng.normal(size=(n, model.dimension)) + mu
    g = model.evaluate(u)
    log_weight = -u @ mu + 0.5 * float(mu @ mu)
    weights = np.exp(np.clip(log_weight, -700.0, 700.0))
    contributions = weights * (g <= 0.0)
    probability = float(np.mean(contributions))
    standard_error = float(np.std(contributions, ddof=1) / np.sqrt(n)) if n > 1 else None
    return Estimate(
        method="IS",
        probability=probability,
        standard_error=standard_error,
        evaluations=n,
        elapsed_seconds=time.perf_counter() - start,
        details={
            "mean_shift_common_factor": float(shift),
            "proposal_events": int(np.sum(g <= 0.0)),
            "effective_sample_size": float(
                np.sum(weights) ** 2 / np.sum(weights**2)
            ),
        },
    )


def subset_simulation(
    model: IEEE24OverloadModel,
    n: int,
    conditional_probability: float,
    max_levels: int,
    proposal_correlation: float,
    rng: np.random.Generator,
) -> Estimate:
    """Au--Beck-style subset simulation for ``g(u) <= 0``.

    The Gaussian proposal

        u' = rho*u + sqrt(1-rho**2)*epsilon

    preserves the standard-normal target.  Rejection of proposals outside
    the current intermediate event therefore gives a simple conditional
    Markov kernel.  The returned standard error is ``None`` because the
    chains are correlated; the level thresholds and acceptance rates are
    returned for diagnostics instead.
    """
    if not 0.0 < conditional_probability < 1.0:
        raise ValueError("conditional_probability must be between zero and one")
    if not 0.0 <= proposal_correlation < 1.0:
        raise ValueError("proposal_correlation must be between zero and one")
    n_seeds = max(1, int(round(conditional_probability * n)))
    chain_length = int(math.ceil(n / n_seeds))
    start = time.perf_counter()
    evaluations = 0
    u = rng.normal(size=(n, model.dimension))
    g = model.evaluate(u)
    evaluations += n
    probability_mass = 1.0
    thresholds: list[float] = []
    acceptance_rates: list[float] = []

    for _level in range(max_levels):
        threshold = float(np.quantile(g, conditional_probability))
        thresholds.append(threshold)
        event = g <= 0.0
        if threshold <= 0.0:
            probability = probability_mass * float(np.mean(event))
            return Estimate(
                method="SS",
                probability=probability,
                standard_error=None,
                evaluations=evaluations,
                elapsed_seconds=time.perf_counter() - start,
                details={
                    "levels": len(thresholds),
                    "thresholds": thresholds,
                    "acceptance_rates": acceptance_rates,
                    "event_fraction_last_level": float(np.mean(event)),
                },
            )

        probability_mass *= conditional_probability
        seed_indices = np.argsort(g)[:n_seeds]
        next_u = np.empty_like(u)
        next_g = np.empty_like(g)
        accepted = 0
        position = 0
        for seed_index in seed_indices:
            current = u[seed_index].copy()
            current_g = float(g[seed_index])
            for step in range(chain_length):
                if step > 0:
                    proposal = proposal_correlation * current + math.sqrt(
                        1.0 - proposal_correlation**2
                    ) * rng.normal(size=model.dimension)
                    proposal_g = float(model.performance(proposal))
                    evaluations += 1
                    if proposal_g <= threshold:
                        current, current_g = proposal, proposal_g
                        accepted += 1
                if position < n:
                    next_u[position] = current
                    next_g[position] = current_g
                    position += 1
                if position == n:
                    break
            if position == n:
                break
        u, g = next_u, next_g
        acceptance_rates.append(accepted / max(1, n - n_seeds))

    probability = probability_mass * float(np.mean(g <= 0.0))
    return Estimate(
        method="SS",
        probability=probability,
        standard_error=None,
        evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - start,
        details={
            "levels": len(thresholds),
            "thresholds": thresholds,
            "acceptance_rates": acceptance_rates,
            "event_fraction_last_level": float(np.mean(g <= 0.0)),
        },
    )


def save_plot(estimates: list[Estimate], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [item.method for item in estimates]
    values = [item.probability for item in estimates]
    errors = [item.standard_error or 0.0 for item in estimates]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, yerr=errors, capsize=4, color=["#648FFF", "#785EF0", "#DC267F"])
    ax.set_yscale("log")
    ax.set_ylabel("Estimated probability of loading >= threshold")
    ax.set_title("IEEE RTS-24 rare-event estimation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000, help="MC sample count")
    parser.add_argument("--is-samples", type=int, default=1000, help="IS sample count")
    parser.add_argument("--subset-samples", type=int, default=1000, help="SS sample count per level")
    parser.add_argument("--subset-p0", type=float, default=0.10, help="SS conditional probability")
    parser.add_argument("--subset-levels", type=int, default=8, help="maximum SS levels")
    parser.add_argument("--subset-rho", type=float, default=0.80, help="SS proposal correlation")
    parser.add_argument("--is-shift", type=float, default=2.5, help="IS shift of common load factor")
    parser.add_argument("--base-load-scale", type=float, default=1.15, help="deterministic IEEE24 load multiplier")
    parser.add_argument("--sigma", type=float, default=0.12, help="lognormal load uncertainty")
    parser.add_argument("--correlation", type=float, default=0.75, help="cross-load correlation")
    parser.add_argument("--threshold", type=float, default=100.0, help="branch-loading threshold in percent")
    parser.add_argument("--seed", type=int, default=20260918, help="random seed")
    parser.add_argument("--output", type=Path, default=Path("outputs/wp2/ieee24_overload.json"))
    parser.add_argument("--no-plot", action="store_true", help="do not write the comparison plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.samples, args.is_samples, args.subset_samples) < 2:
        raise SystemExit("all sample counts must be at least 2")
    net = load_ieee24()
    model = IEEE24OverloadModel(
        net,
        base_load_scale=args.base_load_scale,
        sigma=args.sigma,
        correlation=args.correlation,
        threshold_percent=args.threshold,
    )
    baseline = model.max_loading_percent(np.zeros(model.dimension))
    rng = np.random.default_rng(args.seed)
    estimates = [
        crude_monte_carlo(model, args.samples, rng),
        importance_sampling(model, args.is_samples, args.is_shift, rng),
        subset_simulation(
            model,
            args.subset_samples,
            args.subset_p0,
            args.subset_levels,
            args.subset_rho,
            rng,
        ),
    ]

    payload = {
        "case": "IEEE RTS-24",
        "baseline_max_loading_percent": baseline,
        "threshold_percent": args.threshold,
        "load_model": {
            "base_load_scale": args.base_load_scale,
            "sigma": args.sigma,
            "correlation": args.correlation,
            "latent_dimension": model.dimension,
        },
        "results": [asdict(item) for item in estimates],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.no_plot:
        save_plot(estimates, args.output.with_suffix(".png"))

    print(f"IEEE RTS-24 baseline maximum loading: {baseline:.2f}%")
    print(f"Overload event: maximum loading >= {args.threshold:.2f}%")
    print("method  probability       stderr       evaluations   seconds")
    for item in estimates:
        stderr = "n/a" if item.standard_error is None else f"{item.standard_error:.3g}"
        print(
            f"{item.method:>4}  {item.probability:12.5g}  {stderr:>10}  "
            f"{item.evaluations:11d}  {item.elapsed_seconds:8.2f}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
