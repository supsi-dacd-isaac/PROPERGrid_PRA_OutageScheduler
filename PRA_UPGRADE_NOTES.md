# PRA upgrade notes

## What changed

- Added `pra_psa/time_series_pra.py`, a direct implementation of the requested
  time-indexed Monte Carlo PRA procedure.
- Added distinct overload, undervoltage, overvoltage, non-convergence, and
  component-level risk outputs.
- Added sample-conditioned and weather-conditioned contingency probabilities.
- Added `pra_psa/rospra_time_series.py` to run the existing sequential DC
  cascading model consistently over multiple operating points.
- Consolidated the three demos into `demos/run_pra_comparison.py`; the former
  PRA and ROSPRA filenames are retained as thin entry points, and a dedicated
  two-hour Swissgrid entry point was added.
- Added cross-system and time-varying visualizations.
- Repaired package imports for the contingency/weather/ROSPRA modules.

## Scientific interpretation

The static PRA and ROSPRA outputs are complementary, not interchangeable:

- static PRA: expected constraint-violation severity under fixed preventive
  dispatch;
- ROSPRA: probability-weighted demand not served after sequential DC cascading.

The composite static score uses voltage deviations multiplied by 100 so that
they are expressed as percentage-point deviations alongside loading excess.
This score is appropriate for ranking. The separate physical channels remain
the appropriate quantities for reporting.

Cross-system plots include metrics normalized by mean demand (per 100 MW).
Full N-1 catalogues should be used for final comparisons; truncation is exposed
only as a smoke-test option.

## Validation performed

- 31 relevant unit tests passed, including new tests for probability
  conditioning, reproducible sampling, probability-mass conservation, and
  time/component aggregation.
- End-to-end two-hour smoke runs completed on pandapower IEEE 24 and IEEE 118
  cases for both static PRA and ROSPRA.
- Preventive OPF was verified on IEEE 24 without falling back to PF.
- Weather-conditioned execution was verified with the explicitly labelled
  placeholder weather model.

The supplied archive did not include the NDA-protected Swissgrid network and
hourly profiles, so no Swissgrid numerical result is packaged. The runner fails
clearly in strict Swissgrid mode and never substitutes synthetic Swissgrid data.
