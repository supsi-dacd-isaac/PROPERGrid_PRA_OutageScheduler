"""Run the clustered finite-sample CVaR outage scheduler from the PROPER root."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    # Put the PROPER root before the script directory.  This prevents local
    # modules inside ``scheduler_clustered`` from shadowing top-level packages
    # such as ``PROPER/visualization`` during direct script execution.
    try:
        sys.path.remove(project_root_str)
    except ValueError:
        pass
    sys.path.insert(0, project_root_str)

from scheduler.dataprocess import prepare_data
from scheduler_clustered.clustered_cvar_engine import (
    ClusteredCVaRConfig,
    ClusteredCVaRScheduler,
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
    raw_data = prepare_data(conf_path="./config/conf_IEEE118.json")
    data = augment_for_decomposition(
        raw_data,
        include_all_line_contingencies=True,
        include_generator_contingencies=False,
        prefer_ppc_branch_limits=True,
    )

    data["master_output_flag"] = 0
    data["allow_outage_deferral"] = False
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

    config = ClusteredCVaRConfig(
        max_iterations=40,
        master_time_limit=120.0,
        master_mip_gap=0.01,
        master_threads=8,
        oracle_workers=4,
        representative_state_mode="network_screening",
        representative_candidate_times=12,
        # Use a screened contingency set during search and validate the final
        # schedule against the complete configured N-1 set.
        contingency_top_k=10,
        mandatory_contingencies=(),
        # Tail-stratified empirical resampling preserves complete nodal demand
        # vectors.  Replace/calibrate sigma and scenario generation before a
        # research case study; sigma=0 uses only historical cluster states.
        scenario_count=16,
        scenario_seed=42,
        scenario_sampling_scheme="stratified_tail",
        global_lognormal_sigma=0.05,
        risk_mode="cvar_objective",
        cvar_alpha=0.95,
        cvar_penalty_weight=10.0,
        expected_loss_penalty_weight=0.0,
        risk_proxy_learning_rate=0.40,
        max_candidates=15,
        patience=6,
        full_contingency_validation=True,
        results_path="clustered_cvar_results.json",
    )

    result = ClusteredCVaRScheduler(data, config).solve()
    logger.info("Termination: %s", result.termination_reason)
    logger.info("Best schedule: %s", result.best_schedule)
    logger.info("Deferred outages: %s", result.deferred_outages)
    logger.info("Best score: %s", result.best_score)
    if result.best_risk is not None:
        logger.info(
            "Best finite-sample risk: E[DNS]=%.6g MW, VaR=%.6g MW, CVaR=%.6g MW",
            result.best_risk.expected_loss,
            result.best_risk.var,
            result.best_risk.cvar,
        )


if __name__ == "__main__":
    main()
