"""Run the deterministic outage-cluster scheduler from the PROPER root."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

# Support both ``python -m scheduler_clustered.main_clustered`` and direct
# execution of scheduler_clustered/main_clustered.py from the project root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    # Put the PROPER root before the script dir, prevents modules inside ``scheduler_clustered`` from shadowing
    # such as ``PROPER/visualization`` during direct script execution.
    try:
        sys.path.remove(project_root_str)
    except ValueError:
        pass
    sys.path.insert(0, project_root_str)

from scheduler.dataprocess import prepare_data
from scheduler_clustered.clustered_engine import (ClusteredDeterministicConfig, ClusteredDeterministicScheduler)
from scheduler_clustered.data_adapter import (augment_for_decomposition, power_data_diagnostics)

logging.basicConfig(format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    raw_data = prepare_data(conf_path="./config/conf_IEEE118_v2.json")
    data = augment_for_decomposition(raw_data,
                                     include_all_line_contingencies=True,
                                     include_generator_contingencies=False,
                                     prefer_ppc_branch_limits=True, )
    # Master settings.
    data["master_output_flag"] = 0
    data["allow_outage_deferral"] = False
    data["security_proxy_weight"] = 1.0

    # Joint cluster-SCOPF settings. Replace the uniform corrective range with
    # generator-specific MW values when calibrated data are available.
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

    config = ClusteredDeterministicConfig(
        max_iterations=40,
        master_time_limit=120.0,
        master_mip_gap=0.01,
        master_threads=8,
        # One network-aware operating state is retained per constant-topology
        # cluster. Set to "peak_total_demand" for the simpler load-only rule.
        representative_state_mode="network_screening",
        representative_candidate_times=12,
        # None means the complete configured N-1 contingency set is included
        # in every joint cluster slave. Use a number for screened development
        # runs and retain full_contingency_validation=True.
        contingency_top_k=None,
        security_mode="penalty",  # penalty or budget
        security_penalty_weight=10.0,
        total_dns_component_weight=0.10,
        risk_proxy_learning_rate=0.40,
        maximum_incremental_dns_budget=0.25,
        total_incremental_dns_budget=float("inf"),
        max_candidates=15,
        patience=6,
        full_contingency_validation=True,
        results_path="clustered_deterministic_results.json",
    )

    result = ClusteredDeterministicScheduler(data, config).solve()
    logger.info("Termination: %s", result.termination_reason)
    logger.info("Best schedule: %s", result.best_schedule)
    logger.info("Deferred outages: %s", result.deferred_outages)
    logger.info("Best score: %s", result.best_score)
    logger.info("Deterministic security cost: %s",result.deterministic_security_cost)


if __name__ == "__main__":
    main()
