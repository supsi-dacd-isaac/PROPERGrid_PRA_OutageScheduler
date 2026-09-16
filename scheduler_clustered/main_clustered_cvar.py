"""Run the clustered Gaussian-scenario CVaR outage scheduler."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scheduler_clustered.clustered_cvar_engine import (
    ClusteredCVaRConfig,
    ClusteredCVaRScheduler,
)
from scheduler_clustered.data_adapter import power_data_diagnostics
from scheduler_clustered.master_problem import solution_from_start_times
from scheduler_clustered.runtime import (
    apply_data_overrides_from_json,
    configured_decomposition_data,
    dataclass_config_from_json,
)

logging.basicConfig(
    format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _deterministic_benchmark(data: dict) -> list:
    path = os.getenv("PROPER_DETERMINISTIC_RESULTS")
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        logger.warning("Deterministic benchmark not found: %s", source)
        return []
    document = json.loads(source.read_text(encoding="utf-8"))
    starts = document.get("best_schedule") or {}
    deferred = document.get("deferred_outages") or ()
    if not starts and not deferred:
        logger.warning("Deterministic result does not contain a schedule.")
        return []
    logger.info("Including deterministic schedule as a CVaR benchmark.")
    return [
        solution_from_start_times(
            data,
            starts,
            deferred_outages=deferred,
        )
    ]


def main() -> None:
    network_config = os.getenv(
        "PROPER_SCHEDULER_CONFIG",
        "./config/IEEE24_scheduler_v2.json",
    )
    clustered_config = os.getenv(
        "PROPER_CLUSTERED_CVAR_CONFIG",
        str(Path(__file__).with_name("config_clustered_cvar_example.json")),
    )
    logger.info("Using network configuration: %s", network_config)
    logger.info("Using clustered CVaR configuration: %s", clustered_config)

    data = configured_decomposition_data(
        network_config,
        allow_outage_deferral=True,
    )
    apply_data_overrides_from_json(data, clustered_config)
    diagnostics = power_data_diagnostics(data)
    logger.info("Power-data diagnostics: %s", diagnostics)
    for warning in diagnostics["warnings"]:
        logger.warning("Data diagnostic: %s", warning)

    config = dataclass_config_from_json(
        ClusteredCVaRConfig,
        clustered_config,
        "clustered_cvar",
    )
    result = ClusteredCVaRScheduler(
        data,
        config,
        benchmark_schedules=_deterministic_benchmark(data),
    ).solve()
    logger.info("Termination: %s", result.termination_reason)
    logger.info("Best schedule: %s", result.best_schedule)
    logger.info("Deferred outages: %s", result.deferred_outages)
    logger.info("Best score: %s", result.best_score)
    if result.best_risk is not None:
        logger.info(
            "Best finite-sample risk: E[L]=%.6g MW, VaR=%.6g MW, CVaR=%.6g MW",
            result.best_risk.expected_loss,
            result.best_risk.var,
            result.best_risk.cvar,
        )


if __name__ == "__main__":
    main()
