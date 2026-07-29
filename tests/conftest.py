"""Shared pytest configuration for PROPERGrid.

The default test run is solver- and data-independent. Integration and Gurobi
checks are opt-in because public CI runners may not have the case data or a
solver licence.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests requiring public power-system data and pandapower",
    )
    parser.addoption(
        "--run-gurobi",
        action="store_true",
        default=False,
        help="run tests importing or invoking Gurobi-dependent modules",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-integration"):
        marker = pytest.mark.skip(reason="use --run-integration to enable this test")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(marker)

    if not config.getoption("--run-gurobi"):
        marker = pytest.mark.skip(reason="use --run-gurobi to enable this test")
        for item in items:
            if "gurobi" in item.keywords:
                item.add_marker(marker)


@pytest.fixture
def fake_network() -> SimpleNamespace:
    """Small pandapower-like object for contingency unit tests."""

    class FakeNetwork(SimpleNamespace):
        def deepcopy(self):
            return copy.deepcopy(self)

    return FakeNetwork(
        line=pd.DataFrame(
            {"in_service": [True, False, True]},
            index=pd.Index([0, 2, 4], name="line"),
        ),
        trafo=pd.DataFrame(
            {"in_service": [True]},
            index=pd.Index([1], name="trafo"),
        ),
        gen=pd.DataFrame(
            {"in_service": [True, True]},
            index=pd.Index([3, 5], name="gen"),
        ),
    )
