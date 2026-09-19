"""Run the deterministic clustered outage scheduler from the PROPER root."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scheduler_clustered.clustered_engine import (
    ClusteredDeterministicConfig,
    ClusteredDeterministicScheduler,
)
from scheduler_clustered.data_adapter import power_data_diagnostics
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


def main() -> None:
    network_config = os.getenv(
        "PROPER_SCHEDULER_CONFIG",
        "./config/conf_IEEE24_scheduler_v2.json",
    )
    clustered_config = os.getenv(
        "PROPER_CLUSTERED_CONFIG",
        str(Path(__file__).with_name("config_clustered_example.json")),
    )
    logger.info("Using network configuration: %s", network_config)
    logger.info("Using clustered configuration: %s", clustered_config)

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
        ClusteredDeterministicConfig,
        clustered_config,
        "clustered_deterministic",
    )
    result = ClusteredDeterministicScheduler(data, config).solve()
    logger.info("Termination: %s", result.termination_reason)
    logger.info("Best schedule: %s", result.best_schedule)
    logger.info("Deferred outages: %s", result.deferred_outages)
    logger.info("Best score: %s", result.best_score)
    logger.info(
        "Deterministic security cost: %s",
        result.deterministic_security_cost,
    )


if __name__ == "__main__":
    main()
