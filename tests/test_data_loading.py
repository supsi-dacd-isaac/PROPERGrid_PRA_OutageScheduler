from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pra_psa.data import (
    load_hourly_demand,
    normalize_bus_load_columns,
    read_matpower_m,
)


MATPOWER_CASE = """
function mpc = case_test
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
1 3 0 0 0 0 1 1 0 230 1 1.1 0.9; % slack
2 1 50 10 0 0 1 1 0 230 1 1.1 0.9;
];
mpc.gen = [
1 50 0 100 -100 1 100 1 100 0 0 0 0 0 0 0 0 0 0 0 0;
];
mpc.branch = [
1 2 0.0 0.1 0.0 100 100 100 0 0 1 -360 360;
];
mpc.gencost = [
2 0 0 3 0.01 1 0;
];
"""


@pytest.mark.unit
def test_read_matpower_m_parses_core_tables(tmp_path: Path) -> None:
    path = tmp_path / "System.m"
    path.write_text(MATPOWER_CASE, encoding="utf-8")
    case = read_matpower_m(path)
    assert case["baseMVA"] == 100.0
    assert case["bus"]["bus_i"].tolist() == [1, 2]
    assert case["gen"].iloc[0]["bus"] == 1
    assert case["branch"].iloc[0]["rateA"] == 100.0
    assert list(case["gencost"].columns) == ["model", "startup", "shutdown", "n", "c2", "c1", "c0"]


@pytest.mark.unit
def test_read_matpower_m_requires_base_mva(tmp_path: Path) -> None:
    path = tmp_path / "bad.m"
    path.write_text("mpc.bus = [1 1];", encoding="utf-8")
    with pytest.raises(ValueError, match="baseMVA"):
        read_matpower_m(path)


@pytest.mark.unit
def test_pickle_demand_loader_and_column_normalisation(tmp_path: Path) -> None:
    expected = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["bus_0", "bus_1"])
    expected.to_pickle(tmp_path / "hourlyDemandBus.pkl")
    loaded = load_hourly_demand(tmp_path)
    pd.testing.assert_frame_equal(loaded, expected)

    bus = pd.DataFrame({"bus_i": [10, 20]})
    normalised = normalize_bus_load_columns(loaded, bus)
    assert normalised.columns.tolist() == [10, 20]
    np.testing.assert_allclose(normalised.to_numpy(), expected.to_numpy())


@pytest.mark.unit
def test_load_column_mismatch_is_left_unchanged() -> None:
    loads = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    bus = pd.DataFrame({"bus_i": [1, 2]})
    result = normalize_bus_load_columns(loads, bus)
    assert result.columns.tolist() == ["a", "b", "c"]


@pytest.mark.unit
def test_missing_hourly_demand_returns_none(tmp_path: Path) -> None:
    assert load_hourly_demand(tmp_path) is None
