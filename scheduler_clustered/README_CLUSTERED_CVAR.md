# Clustered deterministic and CVaR outage scheduler

This package contains the deterministic outage-cluster scheduler and a
finite-sample CVaR extension.  Both formulations retain a compact start-index
maintenance master and evaluate network security only after converting a
candidate schedule into maximal constant-topology outage clusters.

## Deterministic formulation

Run:

```bash
python -m scheduler_clustered.main_clustered
```

Each cluster is evaluated at one network-aware representative nodal-demand
state through a soft joint N-1 SCOPF.  The schedule score combines maintenance
utility and duration-weighted incremental DNS relative to the no-maintenance
baseline.

## CVaR formulation

Run:

```bash
python -m scheduler_clustered.main_clustered_cvar
```

For a candidate schedule `y`, let `K(y)` denote its outage clusters.  For each
cluster `k`, nodal-demand scenario `s`, and contingency `c`, the operational
slave computes soft-SCOPF demand not served `DNS[k,s,c]`.  The scenario loss is

```text
L_s(y) = sum_k duration(k)/horizon
         * positive_part(max_c DNS_outage[k,s,c]
                         - max_c DNS_baseline[k,s,c]).
```

The weighted finite-sample CVaR is

```text
CVaR_alpha(L) = eta + 1/(1-alpha) * sum_s p_s xi_s,
xi_s >= L_s - eta,
xi_s >= 0.
```

The candidate objective is

```text
maximize maintenance_utility
         - expected_loss_penalty_weight * E[L]/peak_demand
         - cvar_penalty_weight * CVaR_alpha(L)/peak_demand.
```

The expectation, VaR, and CVaR of every evaluated complete schedule are exact
for the finite scenario set.  To avoid rebuilding the full extensive-form
mixed-integer model, the start-index master uses adaptive linear risk
coefficients learned from the scenarios in the current CVaR tail.  Accordingly,
the complete algorithm is a simulation-optimization matheuristic, not an exact
Benders or extensive-form solution of the stochastic mixed-integer program.

### Scenario model

The built-in `EmpiricalClusterScenarioSampler` resamples complete historical
nodal-demand vectors from each cluster, preserving their spatial dependence.
The default tail-stratified quadrature assigns explicit probability mass to
`[0.95, 1.00]`, allowing meaningful 95% CVaR estimation with a moderate number
of scenarios.  A common scenario quantile is used across clusters, producing a
conservative comonotonic cross-cluster dependence assumption.

`global_lognormal_sigma` applies an optional aggregate multiplier.  It is only a
development mechanism and must be calibrated before publication.  For a
research case study, replace the built-in sampler with externally calibrated
nodal trajectories, preferably preserving spatial and temporal dependence
through a copula, forecast-error model, or historical path resampling.

### Recommended staged settings

IEEE-24 development:

```python
scenario_count=16
cvar_alpha=0.95
contingency_top_k=10
oracle_workers=4
full_contingency_validation=True
```

Final IEEE-24 assessment:

```python
scenario_count=50
cvar_alpha=0.95
contingency_top_k=None
```

IEEE-118 or Swissgrid should use screened contingencies during the search and a
larger independent validation scenario set for the final schedule.

## Visualization

The new visualizer reads either deterministic or CVaR JSON results without
loading Gurobi or the original network data:

```bash
python -m scheduler_clustered.results_visualization clustered_deterministic_results.json
python -m scheduler_clustered.results_visualization clustered_cvar_results.json
```

Equivalent convenience command:

```bash
python visualize_clustered_results.py clustered_deterministic_results.json
```

Add `--show` for interactive windows and `--output-dir PATH` to choose the
output directory.

To compare schedules:

```bash
python -m scheduler_clustered.results_visualization \
    clustered_deterministic_results.json \
    --compare clustered_cvar_results.json
```

The visualizer writes:

- outage schedule heatmap;
- Gantt schedule;
- concurrent-outage trajectory;
- incremental DNS by cluster;
- cluster-severity timeline;
- ranked critical clusters;
- outage-interaction severity matrix;
- critical-contingency ranking;
- outer-loop convergence;
- CVaR scenario-loss distribution and empirical CDF, when available;
- `cluster_summary.csv`, `candidate_summary.csv`, and `scenario_losses.csv`.

## Important input-model checks

The current IEEE-24 run reports one pandapower `ext_grid` that is not represented
in the scheduler generator set and branch limits with only three repeated
values.  Resolve these warnings before interpreting DNS values as a validated
reliability assessment.
