# Revision summary

## Version 2.0.0

- Replaced the master scheduling formulation with exact start-indexed,
  contiguous-duration scheduling and optional outage deferral.
- Added priority-weighted coverage and early-execution utility terms.
- Retained maximal static-topology clustering.
- Added three critical deterministic states per topology cluster.
- Added correlated Gaussian nodal-demand sampling around the same three states.
- Unified deterministic and CVaR evaluation through the soft N-1 DC SCOPF with
  nodal DNS, spillage, and bounded corrective redispatch.
- Added a deterministic-benchmark CVaR guardrail and independent paired risk
  validation on common random scenarios.
- Added caching, optional parallel slave evaluation, diagnostics, configuration
  helpers, tests, and result-visualisation compatibility.
