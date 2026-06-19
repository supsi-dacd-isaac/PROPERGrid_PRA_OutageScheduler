import pandas as pd

from pra_psa.models import Contingency, Outage
from pra_psa.risk import UniformContingencyModel, aggregate_risk


def test_contingency_normalization():
    c = Contingency.from_any({"element_type": "line", "element_index": 4})
    assert c.order == 1
    assert c.contingency_id == "line:4"
    assert c.outages[0] == Outage("line", 4)


def test_risk_aggregation():
    df = pd.DataFrame(
        {
            "time": [0, 0, 1, 1],
            "contingency_id": ["base", "N-1::line:0", "base", "N-1::line:0"],
            "severity": [0.0, 10.0, 1.0, 20.0],
            "converged": [True, True, True, True],
            "n_overloads": [0, 1, 0, 2],
        }
    )
    probs = {"N-1::line:0": 0.01}
    risk = aggregate_risk(df, probabilities=probs)
    assert not risk.risk_by_contingency.empty
    assert risk.risk_by_time["total_risk"].sum() > 0


def test_uniform_probability_model():
    contingencies = [Contingency((Outage("line", 0),)), Contingency((Outage("line", 1),))]
    probs = UniformContingencyModel(0.02).get_probabilities(contingencies)
    assert probs["line:0"] == 0.01
    assert probs["line:1"] == 0.01
