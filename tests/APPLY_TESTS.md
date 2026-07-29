# PROPERGrid repository test suite

Copy the patch into the repository root while preserving paths.

The patch **replaces the current `tests/` directory**. Remove the four stale tests
before copying because they import modules/functions that no longer exist:

```text
tests/test_importance_sampling.py
tests/test_monte_carlo.py
tests/test_risk_and_network.py
tests/test_subset_simulation.py
```

Then copy:

```text
tests/
pytest.ini
.coveragerc
requirements-test.txt
.github/workflows/tests.yml
```

The common-pool source patch must already be present for
`tests/test_common_risk_selection.py`.

## Commands

Install test dependencies:

```bash
pip install -e .
pip install -r requirements-test.txt
```

Run the default solver-independent suite:

```bash
pytest -m "unit and not gurobi"
```

Run public-case integration checks:

```bash
pytest --run-integration -m integration
```

Run Gurobi-dependent import/helper checks:

```bash
pytest --run-gurobi -m gurobi
```

Run everything available on a licensed workstation:

```bash
pytest --run-integration --run-gurobi
```

## Test philosophy

The default suite uses synthetic data and does not require a Gurobi licence. It
covers canonical PRA/PSA models, contingency construction, risk aggregation,
severity metrics, MATPOWER parsing, demand sampling, outage clustering,
scenario sampling, strict JSON serialisation, and common-pool risk selection.

Integration tests are intentionally opt-in because public case files may be
omitted from a checkout. Gurobi-dependent tests are also opt-in.

Known placeholder APIs are marked `xfail` so they remain visible as technical
debt without making routine CI fail.
