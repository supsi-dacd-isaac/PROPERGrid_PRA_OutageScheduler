# Deterministic outage-cluster decomposition

This package implements a compact deterministic master/slave formulation for
PROPER-Grids.

## Mathematical structure

### Master problem

For outage `o` and feasible start period `tau`, the binary variable

```
y[o,tau] = 1
```

selects its starting index. The active-outage expression is derived directly:

```
x[o,t] = sum(y[o,tau] : tau <= t < tau + duration[o])
```

No power-flow, contingency-flow, angle, or load-shedding variables are placed
in the master.

### Outage clusters

After each master solution, consecutive periods with an identical planned
outage set are merged. For example:

| Period | Planned outages |
|---|---|
| June | A |
| July-August | A, B |
| September | B |

This yields three slave problems rather than one slave for every period.

### Representative operating state

One deterministic nodal-demand realization is selected for each cluster.
Two rules are implemented:

- `peak_total_demand`: highest total demand in the cluster;
- `network_screening`: shortlist the highest-load periods and select the one
  with the largest PTDF/LODF post-contingency loading score.

The network-aware rule is the recommended default because maximum total demand
is not necessarily the most severe transmission state.

### Joint soft N-1 SCOPF slave

For the representative state of a cluster, one continuous LP simultaneously
models:

- a common normal-state dispatch;
- every selected N-1 line/generator contingency;
- contingency-specific corrective redispatch;
- line-flow limits and DC power flow;
- nodal load shedding;
- generation spillage;
- one angle reference per connected component.

Load shedding and spillage guarantee complete recourse. The objective gives
priority to minimizing the maximum contingency DNS, followed by total DNS and
operating costs.

### Baseline-relative severity

The same representative state and contingency set are solved without planned
outages. The cluster penalty is based on positive incremental curtailment:

```
max_increment = max(0, candidate.max_DNS - baseline.max_DNS)
total_increment = max(0, candidate.total_DNS - baseline.total_DNS)
```

This prevents pre-existing benchmark insecurity from making the maintenance
problem automatically infeasible.

## Operating modes

### `security_mode="penalty"` — recommended

Every schedule remains feasible because slave load shedding is soft. The
algorithm minimizes a deterministic duration-weighted incremental-security
cost through adaptive linear master penalties and no-good exploration. It
always returns the best evaluated schedule unless the scheduling constraints
themselves are infeasible.

This is a deterministic simulation-optimization matheuristic, not an exact
Benders optimality-cut implementation.

### `security_mode="budget"`

A cluster is accepted only if its incremental maximum and total DNS remain
within configured limits. Violating outage overlaps generate one cluster-level
logic cut at the representative period. This is much more compact than adding
one cut per contingency state.

If no complete schedule satisfies the budgets, enable:

```python
data["allow_outage_deferral"] = True
```

The master then schedules or explicitly defers every outage.

## Installation

Copy `scheduler_clustered` into the PROPER root beside `scheduler`:

```text
PROPER/
├── scheduler/
├── scheduler_clustered/
└── config/
```

Run:

```bash
python -m scheduler_clustered.main_clustered
```

Direct execution also works:

```bash
python scheduler_clustered/main_clustered.py
```

## Recommended IEEE-24 configuration

```python
config = ClusteredDeterministicConfig(
    representative_state_mode="network_screening",
    representative_candidate_times=12,
    contingency_top_k=None,
    security_mode="penalty",
    max_candidates=15,
    patience=6,
)
```

For development on IEEE-118, start with `contingency_top_k=20` and retain
`full_contingency_validation=True`. Increase the screened set after verifying
runtime and memory.

## Outputs

`clustered_deterministic_results.json` includes:

- best outage starts;
- deferred outages, when enabled;
- active outages by period;
- all evaluated schedules;
- outage clusters for each candidate;
- selected representative periods;
- baseline and candidate maximum/total DNS;
- worst contingency in each joint slave;
- deterministic incremental-security cost;
- final full-contingency validation, when screening is used.

## Important data checks

The operational result remains conditional on the input model. Before drawing
physical conclusions, verify:

- `ext_grid` or equivalent slack/import capacity is represented explicitly;
- generator minimum and maximum outputs are physical;
- branch `RATE_A` values are physical rather than heuristic placeholders;
- demand, branch ratings, and generation use consistent MW units;
- corrective redispatch ranges are generator-specific where possible.
