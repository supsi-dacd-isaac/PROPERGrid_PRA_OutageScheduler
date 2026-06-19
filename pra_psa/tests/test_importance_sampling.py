
# tests/test_importance_sampling.py
from pra_psa.simulation.importance_sampling import importance_sample_contingencies

def test_importance_sample_shape():
    cset = [(i,) for i in range(10)]
    probs = [0.1] * 10
    samples, weights = importance_sample_contingencies(cset, probs, 100)

    assert len(samples) == 100
    assert len(weights) == 100
