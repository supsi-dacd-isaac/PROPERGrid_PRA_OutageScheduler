# IEEE RTS-24 fixed-scenario outage scheduling

This directory runs three start-time master searches on the bundled IEEE RTS-24
MATPOWER network with **one shared corrective DC-SCOPF slave**:

```python
study = load_study("my_scenarios.sqlite", "my_config.json")
det = Master_DET(study)
cvar = Master_CVAR(study, det)
dro = Master_DRO_CVAR(study, det)
```

Each call takes a validated `StudyData` object and returns a `ScheduleResult`
with starts (or `null` for explicitly deferrable tasks), utility, scenario losses,
comparison metrics, solver counts, and runtime. The three masters share the
same fixed scenario bank and `SLAVE_DCSCOPF`. DET selects with one declared
representative scenario path; CVaR and DRO-CVaR select among schedules meeting
`U >= U_DET - utility_tolerance`. Candidate proposals use SciPy MILP with
no-good cuts and an updated additive risk proxy. Risk evaluation uses the
actual joint DC-SCOPF LP; **the finite candidate search is a heuristic**, so
the result is not a global optimum over all schedules.

## Quick smoke run

From the repository root, with Python 3.10+ and `pip install -r
ieee24_standalone/requirements.txt`:

```bash
python -m ieee24_standalone.examples.make_synthetic_bank /tmp/ieee24_demo.sqlite
python -m ieee24_standalone.run \
  --database /tmp/ieee24_demo.sqlite \
  --config ieee24_standalone/examples/demo_config.json \
  --output /tmp/ieee24_demo_out
```

The generated bank has three hand-specified paths and three daily time steps.
Its wind, demand, line bearings, and hazard coefficients are **illustrative**, not
measured or calibrated. The demo uses `alpha=0.5` so three scenarios show a
nontrivial tail. Use a substantially larger independent bank for a study at
`alpha=0.95`; 48 equally weighted paths contain only 2.4 effective upper-tail
scenario masses. Output files are `schedules.json` and `comparison.csv`.

## Fixed SQLite bank

Create these four tables before calling `load_study`:

```sql
CREATE TABLE scenario(
  scenario_id TEXT PRIMARY KEY, probability REAL NOT NULL
);
CREATE TABLE time_step(
  step INTEGER PRIMARY KEY, timestamp_utc TEXT UNIQUE NOT NULL,
  duration_h REAL NOT NULL
);
CREATE TABLE demand(
  scenario_id TEXT NOT NULL, step INTEGER NOT NULL, bus_id INTEGER NOT NULL,
  demand_mw REAL NOT NULL, PRIMARY KEY(scenario_id, step, bus_id)
);
CREATE TABLE weather(
  scenario_id TEXT NOT NULL, step INTEGER NOT NULL, line_id INTEGER NOT NULL,
  speed_mps REAL NOT NULL, direction_deg REAL NOT NULL,
  PRIMARY KEY(scenario_id, step, line_id)
);
```

Each `(scenario_id, step)` must contain all 24 bus demands and all 38 branch
wind speed/direction pairs, including branches with zero modeled hazard. Bus
IDs are MATPOWER bus numbers (1–24); line and generator IDs are **zero-based
MATPOWER row indices**, not the pandapower component indices in the legacy
`IEEE24_scheduler_v2.json`. Steps must be `0..H-1`, timestamps strictly
increasing UTC, probabilities nonnegative and summing to one, and directions
in degrees `[0, 360)`. A time step is a planning slot: maintenance durations
and windows are numbers of steps, while `duration_h` enters the weather event
probabilities and horizon weights. The bank is read once and never resampled.

## Configuration

Copy `examples/demo_config.json`. Each task defines `task_id`, `asset_type`
(`line` or `gen`), zero-based `asset_id`, `duration_steps`, `earliest_start`,
`latest_start`, and `utility`. By default every task is required; set
`deferrable: true` to allow a null start with zero utility. Optionally set
`delay_penalty_per_step` for later starts. At most one task per physical asset
is supported. `max_active` limits simultaneous work. Tasks and scenarios are
data, not generated within the optimizer.

For each line with nonzero weather hazard, provide in `hazard` a `bearing_deg`
in `[0,180)`, `base_rate_per_h`, `speed_threshold_mps`,
`speed_slope_per_mps`, and `max_rate_per_h`. Missing hazard entries mean a
zero event rate. Every positive-rate line must be included in
`contingency_lines`. Wind direction is the meteorological azimuth; transverse
exposure uses `speed_mps * abs(sin(direction_deg - bearing_deg))`. Changing the
direction by 180° has the same transverse exposure. Calibration and branch
bearings must be supplied for physical failure-risk claims.

Key `settings`: `alpha`, `tv_radius` in `[0,1]`, `max_candidates`,
`utility_tolerance` (absolute utility units), `loss_scale` (DET's MW-to-utility
weight), `proxy_gain`, `redispatch_fraction` (corrective change as a fraction
of generator Pmax), `deterministic_scenario_id`, and explicit
`contingency_lines`/`contingency_generators`. The defaults screen **all** line
N-1 states and no generator N-1 states. For a large horizon, optionally set
`representatives_per_topology` to 1 or 2: each contiguous outage-topology
interval then uses fixed high-demand/high-wind dates from each scenario path,
weighted by the interval's duration. `0` (default) evaluates every step. A
screened contingency set or representative-step setting changes the modeled
loss; report it alongside results and validate selected schedules with a
complete catalogue before making operational claims.

## Score and limits

The slave minimizes worst-state DNS with a small total-DNS tie-break over the
base state and every enabled, surviving N-1 state. It enforces DC branch flow,
thermal limits, one angle reference per island, base/corrective generator
dispatch and redispatch limits, bus balance, bounded load shedding, and local
generation spill. It uses MATPOWER Pmax and branch ratings, tap, and phase
shift. The DC model omits reactive power, voltage, unit commitment, Pmin,
and generator ramp data; `redispatch_fraction` is a study assumption.

For each dated scenario/step, line event rates are
`min(max_rate, base_rate * exp(slope * max(v_transverse - threshold, 0)))`.
The first-event model assigns no-event mass `exp(-duration_h * sum(rates))`
and splits the remaining mass among available lines in proportion to their
rates. Conditional expected DNS is computed separately for the candidate and
the **paired no-maintenance baseline**, using the same dated demand and wind.
The nonnegative increment is duration-weighted across evaluated steps to
form one loss in MW per scenario. It is a model-based DNS score, **not** EENS,
monetary cost, or realized-event CVaR. Generator contingencies can constrain
the security dispatch but currently have no weather event probability.

The empirical weighted upper CVaR is evaluated directly. DRO-CVaR solves a
finite-support LP for

\[
\sup_{q\ge0,\;\sum q=1,\;\frac12\lVert q-p\rVert_1\le\rho}
\operatorname{CVaR}_{\alpha,q}(L(y)).
\]

This reweights only supplied paths; it does not represent storms absent from
the bank. The comparison CSV reports the same bank-based metrics for all
three selected schedules. Independent held-out paths and full contingency
validation are needed for claims beyond in-sample sensitivity.

## Tests

```bash
python -m unittest discover -s ieee24_standalone/tests -v
```

The packaged `case/System.m` is copied from this repository's
`data/powersystems/IEEE24/System.m`; its attribution and IEEE RTS references
remain in the file header.
