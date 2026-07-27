"""Utility tests that do not require Gurobi or pandapower."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weighted_cvar_atom():
    risk = load_module("risk_metrics_standalone", "risk_metrics.py")
    result = risk.weighted_var_cvar(
        {"normal": 0.0, "moderate": 10.0, "extreme": 100.0},
        {"normal": 0.80, "moderate": 0.15, "extreme": 0.05},
        alpha=0.90,
    )
    assert np.isclose(result.var, 10.0)
    assert np.isclose(result.cvar, 55.0)


def test_triangle_ptdf_lodf():
    network = load_module("network_model_standalone", "network_model.py")
    cft = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    data = {
        "names": {
            "buses": ["bus_0", "bus_1", "bus_2"],
            "lines": ["line_0", "line_1", "line_2"],
            "generators": ["gen_0"],
        },
        "S": cft.T,
        "B_mat": cft,
        "f_lim": {"line_0": 10.0, "line_1": 10.0, "line_2": 10.0},
        "p_min": {"gen_0": 0.0},
        "p_max": {"gen_0": 10.0},
        "g2bus": ["bus_0"],
        "ref_buses": ["bus_0"],
    }
    model = network.NetworkModel(data)
    result = model.sensitivities((0, 1, 2))
    assert result.ptdf.shape == (3, 3)
    assert result.lodf.shape == (3, 3)
    assert not result.islanding_contingencies
    assert np.allclose(np.diag(result.lodf), -1.0)
