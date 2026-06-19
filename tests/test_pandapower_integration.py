import importlib.util
from pathlib import Path

import pytest

from pra_psa import build_n1_contingencies, load_system, run_contingency_analysis

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

pytestmark = pytest.mark.skipif(importlib.util.find_spec("pandapower") is None, reason="pandapower not installed")


def test_ieee24_n1_contingency_analysis_smoke():
    system = load_system(DATA_DIR, "IEEE24", as_pandapower=True)
    contingencies = build_n1_contingencies(system.net, include=("line",))[:3]
    result = run_contingency_analysis(
        system.net,
        operating_points=system.load_time_series.iloc[:2],
        contingencies=contingencies,
        pf_mode="dc",
        include_base_case=True,
    )
    assert len(result.summary) == 2 * (1 + len(contingencies))
    assert "severity" in result.summary.columns
