# Compact comparative PRA demos

The former scripts implemented three partially overlapping workflows. The
replacement uses one runner and three thin entry points:

```bash
# Static contingency PRA on IEEE 24 and IEEE 118
python demos/run_PRA_contingency_analysis.py \
  --cases ieee24 ieee118 --hours 2 --samples 50

# Cascading/DNS risk using ROSPRA
python demos/run_rospra_demo.py \
  --cases ieee24 ieee118 --hours 2 --samples 50

# Both methods on the NDA-protected Swissgrid case
python demos/run_swissgrid_2h_risk_assessment.py --samples 50

# All cases and both methods
python demos/run_pra_comparison.py \
  --cases ieee24 ieee118 swissgrid --hours 2 --samples 50
```

Use `--max-contingencies 20` only for a smoke test. Omitting it evaluates the
complete selected N-1 line/transformer catalogue. Optional sampled N-2 events
can be added with `--n2-contingencies COUNT`.

## Algorithm correspondence

For every operating point, the static engine:

1. applies the baseline state and obtains a preventive OPF dispatch;
2. samples correlated load states around that baseline;
3. freezes the reference non-slack generation dispatch;
4. evaluates the base state and every selected N-1/N-k state by AC or DC PF;
5. evaluates sample-conditioned contingency probabilities;
6. records overload, undervoltage, overvoltage, non-convergence, and a ranking
   score; and
7. aggregates results by time, contingency, and monitored component.

The ROSPRA path is intentionally separate. It performs sequential DC cascading
simulation and reports probability-weighted demand not served. The static PRA
score and ROSPRA DNS have different definitions and units and are never added.

## Weather extension

Add `--weather` to activate a two-regime weather-conditioned Poisson model. If
`--weather-csv` is omitted, the run is labelled as using a placeholder weather
model. To estimate regime transitions from data:

```bash
python demos/run_pra_comparison.py --weather \
  --weather-csv data/weather/hourly_wind.csv \
  --weather-wind-column wind_speed_m_s
```

The rate values in the compact runner are demonstrative defaults. Replace them
with asset-specific calibrated rates before operational interpretation.

## Outputs

`outputs/pra_comparison/` contains:

- `case_comparison.csv` and `case_comparison.png`;
- `time_varying_risk.csv` and `time_varying_risk.png`;
- per-case `contingency/` tables for time, contingency, component, scenario,
  and sampled probability results; and
- per-case `rospra/` tables for time-indexed DNS and cascading scenarios.

The runner uses repository load profiles when available. If IEEE benchmark
profiles are absent, it uses a labelled synthetic daily scaling. It never
substitutes synthetic profiles for the Swissgrid study.
