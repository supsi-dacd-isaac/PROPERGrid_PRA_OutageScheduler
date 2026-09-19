"""Run the optional common-pool expected-loss versus CVaR comparison."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scheduler_clustered.clustered_cvar_engine import ClusteredCVaRConfig
from scheduler_clustered.common_risk_comparison import (
    CommonRiskComparisonConfig,
    CommonRiskComparisonScheduler,
)
from scheduler_clustered.data_adapter import power_data_diagnostics
from scheduler_clustered.runtime import configured_decomposition_data

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
    data = configured_decomposition_data(
        network_config, allow_outage_deferral=True
    )
    diagnostics = power_data_diagnostics(data)
    logger.info("Power-data diagnostics: %s", diagnostics)
    for warning in diagnostics["warnings"]:
        logger.warning("Data diagnostic: %s", warning)

    risk_config = ClusteredCVaRConfig(
        master_time_limit=120.0,
        master_mip_gap=0.0,
        master_threads=8,
        oracle_workers=4,
        representative_state_mode="network_screening",
        representative_candidate_times=12,
        critical_state_count=3,
        contingency_top_k=10,
        scenario_model="gaussian_critical_states",
        scenario_count=64,
        gaussian_samples_per_critical_state=None,
        scenario_seed=42,
        gaussian_relative_sigma=0.05,
        gaussian_global_sigma=0.03,
        cvar_alpha=0.95,
        full_contingency_validation=True,
        cvar_penalty_weight=0.0,
        expected_loss_penalty_weight=0.0,
    )
    comparison_config = CommonRiskComparisonConfig(
        candidate_pool_size=50,
        relative_utility_tolerance=0.01,
        absolute_utility_tolerance=0.0,
        full_contingency_validation=True,
        results_path=(
            "outputs/cluster_scheduler/common_risk_comparison_results.json"
        ),
    )

    result = CommonRiskComparisonScheduler(
        data,
        risk_config=risk_config,
        comparison_config=comparison_config,
    ).solve_common_pool()
    logger.info("Same selected schedule: %s", result.same_schedule)
    logger.info(
        "Risk-neutral candidate=%d, utility=%.6g, E[L]=%.6g, CVaR=%.6g",
        result.risk_neutral.candidate.pool_index,
        result.risk_neutral.candidate.selection_utility,
        result.risk_neutral.validation_risk.expected_loss,
        result.risk_neutral.validation_risk.cvar,
    )
    logger.info(
        "CVaR candidate=%d, utility=%.6g, E[L]=%.6g, CVaR=%.6g",
        result.cvar.candidate.pool_index,
        result.cvar.candidate.selection_utility,
        result.cvar.validation_risk.expected_loss,
        result.cvar.validation_risk.cvar,
    )


if __name__ == "__main__":
    main()
