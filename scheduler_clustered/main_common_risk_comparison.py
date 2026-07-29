"""Run the controlled common-pool expected-loss versus CVaR comparison."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    try:
        sys.path.remove(project_root_str)
    except ValueError:
        pass
    sys.path.insert(0, project_root_str)

from scheduler.dataprocess import prepare_data
from scheduler_clustered.clustered_cvar_engine import ClusteredCVaRConfig
from scheduler_clustered.common_risk_comparison import (
    CommonRiskComparisonConfig,
    CommonRiskComparisonScheduler,
)
from scheduler_clustered.data_adapter import (
    augment_for_decomposition,
    power_data_diagnostics,
)

logging.basicConfig(
    format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    raw_data = prepare_data(conf_path="./config/conf_IEEE118_v2.json")
    data = augment_for_decomposition(
        raw_data,
        include_all_line_contingencies=True,
        include_generator_contingencies=False,
        prefer_ppc_branch_limits=True,
    )

    data["master_output_flag"] = 0
    data["allow_outage_deferral"] = False
    # The common pool must not receive formulation-specific learned penalties.
    data["security_proxy_weight"] = 1.0
    data["corrective_redispatch_fraction"] = 0.20
    data["redispatch_cost"] = 10.0
    data["cluster_max_dns_weight"] = 1e6
    data["cluster_total_dns_weight"] = 1e3
    data["cluster_generation_cost_weight"] = 1e-3
    data["cluster_oracle_threads"] = 1
    data["oracle_isolated_env"] = True

    diagnostics = power_data_diagnostics(data)
    logger.info("Power-data diagnostics: %s", diagnostics)
    for warning in diagnostics["warnings"]:
        logger.warning("Data diagnostic: %s", warning)

    risk_config = ClusteredCVaRConfig(
        master_time_limit=120.0,
        # Exact master solves preserve the ordering of the common candidate pool.
        master_mip_gap=0.0,
        master_threads=8,
        oracle_workers=4,
        representative_state_mode="network_screening",
        representative_candidate_times=12,
        # The same screened set is used for every candidate during pool
        # evaluation. Both selected schedules are subsequently validated with
        # the complete configured N-1 contingency set.
        contingency_top_k=10,
        mandatory_contingencies=(),
        scenario_count=64,
        scenario_seed=42,
        scenario_sampling_scheme="stratified_tail",
        global_lognormal_sigma=0.05,
        cvar_alpha=0.95,
        full_contingency_validation=True,
        # The fields below are not used for pool selection because no proxy
        # update or weighted score is applied in solve_common_pool().
        cvar_penalty_weight=0.0,
        expected_loss_penalty_weight=0.0,
    )
    comparison_config = CommonRiskComparisonConfig(
        candidate_pool_size=50,
        relative_utility_tolerance=0.01,
        absolute_utility_tolerance=0.0,
        full_contingency_validation=True,
        results_path="outputs/cluster_scheduler/common_risk_comparison_results.json",
    )

    result = CommonRiskComparisonScheduler(
        data,
        risk_config=risk_config,
        comparison_config=comparison_config,
    ).solve_common_pool()

    logger.info("Same selected schedule: %s", result.same_schedule)
    logger.info(
        "Risk-neutral: candidate=%d, utility=%.6g, E[L]=%.6g, CVaR=%.6g",
        result.risk_neutral.candidate.pool_index,
        result.risk_neutral.candidate.selection_utility,
        result.risk_neutral.validation_risk.expected_loss,
        result.risk_neutral.validation_risk.cvar,
    )
    logger.info(
        "CVaR: candidate=%d, utility=%.6g, E[L]=%.6g, CVaR=%.6g",
        result.cvar.candidate.pool_index,
        result.cvar.candidate.selection_utility,
        result.cvar.validation_risk.expected_loss,
        result.cvar.validation_risk.cvar,
    )


if __name__ == "__main__":
    main()
