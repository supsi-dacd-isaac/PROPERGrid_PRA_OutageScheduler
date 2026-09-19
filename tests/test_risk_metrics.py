from scheduler_clustered.risk_metrics import weighted_var_cvar


def test_weighted_cvar_uses_fractional_var_atom():
    losses = {"a": 0.0, "b": 10.0, "c": 20.0}
    probabilities = {"a": 0.8, "b": 0.15, "c": 0.05}
    result = weighted_var_cvar(losses, probabilities, alpha=0.9)
    assert result.var == 10.0
    assert abs(result.cvar - 15.0) < 1e-12
