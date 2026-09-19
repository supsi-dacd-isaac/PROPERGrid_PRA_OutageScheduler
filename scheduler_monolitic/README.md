# Monolithic risk-informed outage scheduler

This folder replaces the previous duplicated deterministic and CVaR implementations with one common stochastic security-constrained outage-scheduling model.

## Main methodological changes

Both schedules now use exactly the same:

- outage variables and maintenance constraints;
- demand-scenario bank and probabilities;
- preventive and corrective dispatch representation;
- DC network model;
- planned-outage topology;
- retained N−1 contingency set;
- system-level schedule loss.

For demand scenario \(s\), the loss is

\[
L_s(y)=\sum_{t\in\mathcal T}w_t\max_{c\in\{0\}\cup\mathcal C}
\operatorname{DNS}_{t,c,s}(y).
\]

The risk-neutral formulation lexicographically minimises expected loss, operating effort, and then maximises maintenance utility. The CVaR formulation minimises

\[
\operatorname{CVaR}_{\beta}(L)=\eta+
\frac{1}{1-\beta}\sum_s p_s\xi_s,
\qquad
\xi_s\ge L_s-\eta,
\]

followed by expected loss and maintenance utility. During the standard comparison, the risk-neutral maintenance utility is imposed as a floor in the CVaR model. The risk-neutral schedule therefore remains feasible for the CVaR optimisation, making the tail-risk comparison explicit.

## Folder structure

```text
optimizers/
├── config.py                       # Solver, model, scenario and risk settings
├── data_adapter.py                 # Legacy DATA conversion and validation
├── scenario_generation.py          # Common demand-scenario banks
├── risk_measures.py                 # Weighted VaR, CVaR and bootstrap tools
├── monolithic_scos.py               # Unified Gurobi formulation
├── postprocessing.py                # Result extraction and persistence
├── schedule_validation.py           # Independent paired validation
├── comparison_visualization.py      # Compact schedule and tail-risk figures
├── run_comparison_optimizer.py      # Main comparison command
├── run_optimizer.py                 # Single-formulation command
├── gurobi_SCOS.py                   # Compatibility wrapper
├── gurobi_SCOS_cvar.py              # Compatibility wrapper
├── legacy/                          # Original files, unchanged
└── tests/                           # Solver-independent unit tests
```

## Recommended comparison run

Run from the PROPER repository root:

```powershell
python -m scheduler_monolitic.run_comparison_optimizer `
  --config config/conf_IEEE24_scheduler.json `
  --training-scenarios 20 `
  --validation-scenarios 100 `
  --beta 0.95 `
  --contingency-top-k 10 `
  --validation-contingency-top-k 10 `
  --validation-batch-size 10 `
  --threads 8 `
  --time-limit 3600
```

For a publication experiment, increase the training bank to at least 50–100 scenarios and the independent validation bank to 500 or more scenarios, subject to computational limits.

Outputs are written under:

```text
outputs/schedule_results/monolitic_scheduler/run_<timestamp>/
```

Figures are written to `outputs/schedule_results/monolitic_scheduler/figures`, while numerical results remain in the timestamped run folder. The principal evidence is `comparison_03_tail_risk`, which contains the paired ECDF, exceedance curve, risk metrics and scenario-wise loss differences. `comparison_04_tail_contingencies` compares N−1 curtailment exposure within each schedule's upper-tail scenarios.

## Interpretation

The decisive metric is

```text
CVaR(CVaR-informed schedule) - CVaR(risk-neutral schedule)
```

A negative value means that the CVaR-informed schedule has the smaller out-of-sample upper-tail consequence. The paired validation summary also reports a bootstrap interval for this difference.

## Computational warning

The complete stochastic model scales approximately with

```text
number of time steps × number of scenarios × number of contingencies.
```

Begin with a screened contingency set and a small training bank. Use a larger independent bank for validation; validation is solved in fixed-schedule scenario batches to limit memory use. For larger transmission systems, the clustered scheduler remains preferable.

## Legacy modules

The original uploaded implementation is preserved unchanged under `legacy/`. Root-level compatibility wrappers retain the former `deterministic_SCOS_gurobi` and `CVAR_SCOS_gurobi` entry points, but new development should use `monolithic_scos.solve_scos`.

## References

- M. Rocha, M. F. Anjos, and M. Gendreau, *Scheduling maintenance with uncertain duration on power transmission systems*, 2023.
- J. Liu, M. Kazemi et al., *Security-Constrained Optimal Scheduling of Transmission Outages With Load Curtailment*, IEEE Transactions on Power Systems, 33(1), 2018.
