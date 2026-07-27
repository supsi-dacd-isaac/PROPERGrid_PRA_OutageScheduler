# Migration from `scheduler_decomposed` v2

The clustered scheduler is installed as a separate package and does not
replace the previous implementation.

## Copy

Copy:

```text
scheduler_clustered/
```

into the PROPER project root, beside `scheduler/` and `scheduler_decomposed/`.

## Run

```bash
python -m scheduler_clustered.main_clustered
```

## Main conceptual differences

| Previous decomposed v2 | Clustered deterministic |
|---|---|
| Evaluates every outage-affected period | Merges periods with identical planned topology |
| Separate base dispatch then contingency LP | One joint SCOPF with a common preventive dispatch |
| One cut per violating time/contingency state | At most one overlap cut per unacceptable cluster |
| Zero/incremental DNS used mainly as feasibility condition | DNS is a deterministic soft recourse cost by default |
| Could eliminate all schedules through cut accumulation | Penalty mode always retains the best evaluated schedule |
| CVaR outer loop | No stochastic scenarios or CVaR |

## Existing data interface

The new package reuses `scheduler.dataprocess.prepare_data()` and the same
augmented data fields used by the v2 decomposition package.
