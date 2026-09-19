# Clustered deterministic and CVaR outage scheduling

This package implements a master/slave scheduler for planned transmission and
generation outages. It compares:

1. a deterministic schedule evaluated at the three most critical operating
   periods of every static-topology cluster; and
2. a CVaR schedule evaluated with correlated Gaussian nodal-demand scenarios
   centred on those critical periods.

The same DC preventive/corrective N-1 SCOPF, contingency catalogue, load
shedding variables, and no-maintenance baseline are used in both formulations.

## 1. Static-topology clusters

For a complete schedule, consecutive periods with the same active planned
outage set are merged into maximal clusters. Empty/no-maintenance intervals are identified by the same routine but omitted
from slave evaluation by default because their incremental maintenance risk is
exactly zero. Set `include_empty_clusters=true` when a complete chronology is
needed in the result file. For example:

- outage A: May--October;
- outage B: August--November;

produces the topology sequence:

- no outage;
- A;
- A+B;
- B;
- no outage.

## 2. Master problem

For outage `o`, `y[o,s] = 1` means that the outage starts in admissible period
`s`. A selected outage occupies exactly `d[o]` consecutive periods. Optional
`defer[o]` variables allow the master to select a feasible subset.

The maintenance utility contains two explicit terms:

- **coverage utility**: rewards completion of high-priority outages;
- **timing utility**: rewards earlier completion, weighted by outage priority.

The master maximises

```text
maintenance utility - learned cluster-DNS proxy - deferral penalties.
```

The exact slave cost is used to rank complete candidate schedules. The linear
proxy only guides the next master iteration.

Implemented constraints include:

- schedule exactly once or defer;
- uninterrupted execution and exact duration through start variables;
- admissible start windows;
- maximum simultaneous outages, globally or by period;
- resource capacities;
- pairwise or group incompatibilities;
- precedence and lag constraints;
- security conflict cuts and no-good exploration cuts.

Useful data fields are:

```python
data["priority"]                         # outage -> non-negative weight
data["outage_start_windows"]             # outage -> [first_start, last_start]
data["allow_outage_deferral"] = True
data["outage_coverage_reward"] = 1.0
data["outage_priority_timing_reward"] = 0.1
data["default_defer_penalty"] = 0.0
data["max_tasks"]                        # scalar or period -> capacity
data["resource_capacities"]
data["resource_usage"]
data["incompatible_outage_groups"]
data["precedence"]
```

Scalar master/oracle settings can be placed directly in the clustered JSON
under `data_overrides`. Case-specific mappings such as `priority`,
`outage_start_windows`, and period-dependent `max_tasks` may be supplied there
as well; they are merged with the values produced by the PROPER data loader and
validated before optimisation.

## 3. Deterministic slave

For each constant-topology cluster, the selector retains `q=3` distinct
network-critical periods. Each period is evaluated under:

- the planned-outage topology;
- the normal state;
- configured N-1 line/generator contingencies;
- DC line-flow equations and thermal limits;
- nodal power balance;
- generator bounds;
- bounded corrective redispatch;
- nodal demand-not-served and generation-spillage recourse.

The SCOPF objective minimises maximum DNS, summed nodal DNS, spillage,
redispatch, and a small normal-generation term. Cluster severity combines the
maximum incremental DNS and mean summed incremental nodal DNS relative to a
no-maintenance baseline at the same operating state.

## 4. CVaR slave

The default scenario model is `gaussian_critical_states`. The parameter `gaussian_samples_per_critical_state` specifies `n` Gaussian
samples for each of the three critical-state indices; the total scenario count
is therefore `3n`.
Spatial correlation is estimated from historical relative nodal-demand
variations and shrunk toward the identity for stability. Common Gaussian and
antithetic draws are reused for every candidate schedule.

The finite-sample schedule loss is

```text
sum_k duration_weight[k] *
    (incremental_max_DNS[k,omega]
     + w_total * mean_incremental_summed_nodal_DNS[k,omega]).
```

Expectation, VaR, and CVaR are computed with the exact scenario probabilities,
including fractional probability mass at the VaR atom.

## 5. Risk-reduction guardrail

`run_cluster_comparison.py` runs the deterministic scheduler first and supplies
its selected schedule to the CVaR scheduler as a benchmark. With
`selection_rule="utility_constrained_cvar"`, the benchmark is included in the
CVaR candidate set. Therefore, among schedules within the configured utility
tolerance, the selected CVaR schedule cannot have higher **optimisation-sample
CVaR** than the deterministic benchmark.

This is not an out-of-sample guarantee. The paired validator evaluates both
final schedules on one independent common Gaussian sample bank and reports
expectation, VaR, CVaR, maximum loss, probability of positive DNS, paired
sample outcomes, and first-order stochastic dominance.

## 6. Running

From the PROPER repository root:

```bash
python -m scheduler_clustered.run_cluster_comparison \
  --config config/conf_IEEE24_scheduler_v2.json
```

Environment variables:

```text
PROPER_SCHEDULER_CONFIG          network/data configuration
PROPER_CLUSTERED_CONFIG          deterministic scheduler JSON
PROPER_CLUSTERED_CVAR_CONFIG     CVaR scheduler JSON
```

Standalone runs:

```bash
python -m scheduler_clustered.main_clustered
python -m scheduler_clustered.main_clustered_cvar
```

The CVaR standalone run can include an existing deterministic benchmark with:

```text
PROPER_DETERMINISTIC_RESULTS=clustered_deterministic_results.json
```

## 7. Calibration recommendations

Before reporting research results, calibrate the Gaussian relative/global
standard deviations from forecast residuals rather than treating the example
values as empirical estimates. Use the complete N-1 set for final validation,
keep scenario seeds fixed for paired comparisons, and perform a separate
out-of-sample or rolling-origin validation.

## 8. Three-axis scalability evidence

The comparison driver exposes `main_scalability()` and the `scalability`
subcommand. It varies (i) network size, using IEEE-24 and IEEE-118, (ii) the
number of planned outages, and (iii) the horizon length. Solver, scenario and
candidate budgets remain fixed across points.

```bash
python -m scheduler_clustered.run_cluster_comparison scalability \
  --grid-case IEEE24=config/conf_IEEE24_scheduler_v2.json \
  --grid-case IEEE118=config/conf_IEEE118_scheduler_v2.json \
  --outage-counts 2 4 8 \
  --horizons 12 24 48 \
  --repeats 3 \
  --deterministic-config scheduler_clustered/config_clustered_example.json \
  --cvar-config scheduler_clustered/config_clustered_cvar_example.json
```

By default the study times optimisation only: figures and paired risk
validation are skipped at every scaling point. This is deliberate because the
reviewer comment concerns scheduler scalability. Use
`--validation-samples 96 --validation-workers 4` only when validation runtime
must also be measured.

The runner preserves every run and writes `scalability_results.json`, raw and
aggregated CSV files, a LaTeX table, a runtime figure,
`scalability_report.md`, and `reviewer_response.md`. It records dimensions,
candidate/scenario budgets, wall-clock times, best-effort memory, termination
status, software/hardware metadata and the exact selected outage list.

For development, `--profile fast` reduces candidates, scenarios, critical
states and screened contingencies. Fast-profile results are automatically
marked non-evidentiary. For final results use the default `evidence` profile,
prefer at least three repeats, and keep identical settings across grid cases.

Every comparison run also writes `computational_performance.json`. The output
records whether the deterministic schedule was included in the CVaR candidate
pool; this must be `true` for the stated in-sample risk guardrail. The driver
now stops when it is false. `--allow-missing-benchmark` exists only to inspect
old, non-evidentiary runs.

The generated reviewer wording is guarded: it becomes favourable only when
IEEE-118 and a comparison grid complete under matched evidence-profile settings
and the deterministic schedule is present in every CVaR candidate pool. Even
then the conclusion is bounded to demonstrated execution on IEEE-118; empirical
log--log slopes are descriptive and are not complexity proofs.

## 9. Runtime reductions

- `--skip-plots --skip-validation` isolates scheduler time.
- Increase `oracle_workers` while keeping `cluster_oracle_threads=1` to avoid
  nested thread oversubscription.
- Use `contingency_top_k` during candidate search, but retain the complete N-1
  catalogue for the final evidentiary validation.
- The deterministic, CVaR and paired-validation caches coalesce simultaneous
  requests for identical SCOPF states, avoiding duplicate solves.
- Use `--reuse-point POINT=RUN_DIR` to resume an interrupted benchmark without
  rerunning completed points.

## Full-horizon paired Monte Carlo N-1 validation

Evaluate the deterministic and CVaR schedules over every planning time step and
all valid N-1 contingencies using common, temporally and spatially correlated
Gaussian load trajectories:

```bash
python -m scheduler_clustered.main_annual_mc_evaluation \
  --config config/conf_IEEE24_scheduler_v2.json \
  --results scheduler_clustered/outputs/cluster_scheduler/common_risk_comparison_results.json \
  --mc-config scheduler_clustered/config_annual_mc_example.json \
  --years 20 \
  --workers 1
```

For separate deterministic and CVaR result files, pass the deterministic file to
`--results` and the CVaR file to `--cvar-results`.

Outputs are written to:

```text
scheduler_clustered/outputs/cluster_scheduler/mc_evaluation/run_<timestamp>/
├── logs/
├── results/
│   ├── hourly_mc_records.csv
│   ├── annual_mc_summary.csv
│   └── annual_mc_risk_summary.json
├── visuals/
└── run_manifest.json
```

The annual sums are worst-contingency DNS exposure metrics. They should not be
called EENS unless the time step is hourly and contingency occurrence
probabilities are explicitly included.

## Run the annual Monte Carlo evaluator from PyCharm

Open `scheduler_clustered/run_annual_mc_evaluation.py`, edit the paths and run controls in the `USER SETTINGS` block, and press **Run**. No terminal arguments are required.

Use `INPUT_MODE = "common"` for `common_risk_comparison_results.json`, or `INPUT_MODE = "separate"` for distinct deterministic and CVaR result files.
