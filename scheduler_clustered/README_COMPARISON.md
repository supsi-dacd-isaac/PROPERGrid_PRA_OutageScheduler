# Clustered deterministic–CVaR comparison

Copy the following items into the PROPER project root:

```text
PROPER/
├── run_clustered_comparison.py
├── visualize_clustered_results.py
└── scheduler_clustered/
```

The package deliberately contains no `scheduler_clustered/visualization.py` file. This avoids shadowing the existing top-level `PROPER/visualization/` package.

## Run both optimizations and create all visuals

From the PROPER root:

```bash
python run_clustered_comparison.py
```

Equivalent module execution:

```bash
python -m scheduler_clustered.run_cluster_comparison
```

The runner uses the current Python interpreter, so a PyCharm configuration should use the same `.venv` that provides Gurobi and pandapower. Set the working directory to the PROPER root.

## Recreate visuals from existing JSON results

```bash
python run_clustered_comparison.py --reuse-results
```

Explicit paths are supported:

```bash
python run_clustered_comparison.py \
  --reuse-results \
  --deterministic-results path/to/clustered_deterministic_results.json \
  --cvar-results path/to/clustered_cvar_results.json
```

## Output structure

Each run creates:

```text
clustered_comparison_runs/run_YYYYMMDD_HHMMSS/
├── logs/
├── results/
├── run_manifest.json
└── visuals/
    ├── deterministic/
    ├── cvar/
    └── comparison/
```

The comparison folder contains:

1. schedule-difference heatmap;
2. outage-start shifts;
3. maintenance-concurrency comparison;
4. normalized search-progress comparison;
5. cluster-DNS empirical CDF comparison;
6. ranked critical-cluster comparison;
7. critical-contingency frequency comparison;
8. outage-interaction severity difference;
9. cluster-duration versus severity comparison;
10. CSV and JSON summary files.

Raw deterministic and CVaR objective scores are not plotted on the same scale because they represent different objective functions. Search progress is normalized within each formulation, while security comparisons use common DNS units.

## Visualize one result file

```bash
python visualize_clustered_results.py clustered_deterministic_results.json
python visualize_clustered_results.py clustered_cvar_results.json
```

or:

```bash
python -m scheduler_clustered.results_visualization clustered_cvar_results.json
```
