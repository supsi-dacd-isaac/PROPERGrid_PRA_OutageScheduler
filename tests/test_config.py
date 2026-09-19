from pathlib import Path

import pytest

from optimizers.config import ModelConfig, RiskConfig, ScenarioConfig, SolverConfig


def test_valid_defaults() -> None:
    SolverConfig().validate()
    ScenarioConfig().validate()
    RiskConfig().validate()
    ModelConfig().validate()


def test_invalid_beta() -> None:
    with pytest.raises(ValueError):
        RiskConfig(beta=1.0).validate()


def test_invalid_contingency_count() -> None:
    with pytest.raises(ValueError):
        ModelConfig(contingency_top_k=0).validate()
