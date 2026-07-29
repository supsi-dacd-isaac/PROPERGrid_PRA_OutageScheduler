# PROPERGrid test plan

## Default unit suite

The default suite is independent of external grid files and a Gurobi licence.
It verifies:

- canonical outage and contingency data models;
- N-1/N-k catalogue construction and contingency application;
- probability normalisation and exact weighted finite-sample CVaR;
- contingency-probability and PRA aggregation models;
- overload, voltage, ENS, and combined severity metrics;
- MATPOWER parsing and nodal-demand loading;
- probabilistic demand sampling and reproducibility;
- constant-topology outage-cluster construction;
- common-random-number cluster scenario sampling;
- strict JSON serialisation;
- common-pool risk-neutral versus CVaR selection.

## Optional suites

`integration` tests use the public IEEE RTS-24 case and pandapower. They are
skipped unless `--run-integration` is specified.

`gurobi` tests check scheduling helper contracts and the common-comparison
result schema. They are skipped unless `--run-gurobi` is specified and
`gurobipy` is installed.

## Known limitations

Placeholder subset-simulation and unified scheduling APIs are represented by
strict `xfail` tests. This records the missing implementation without hiding it
or breaking routine CI.

## Acceptance criteria

A pull request should satisfy:

```bash
pytest -m "unit and not gurobi"
```

A release candidate should additionally satisfy, on a configured workstation:

```bash
pytest --run-integration --run-gurobi
```
