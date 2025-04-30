"""
Centralized configuration for Gurobi optimization parameters.
These parameters are used across all optimization models to ensure consistency.
"""
TimeLimit = 300  # 5 minutes max for standard optimization
STANDARD_PARAMS = {
    'MIPGap': 0.01,      # Allow larger optimality gap
    'Heuristics': 0.8,   # More effort on heuristics
    'Cuts': 1,           # Moderate cut generation
    'Presolve': 1,       # Moderate presolve
    'Threads': 4,        # Use 4 threads (adjust as needed)
    'TimeLimit': TimeLimit,    # 5 minutes max
    'Method': 2,         # Use barrier (for LP) or dual simplex (fast in practice)
    'OutputFlag': 1,     # Show basic solver output
}
# Parameters for deterministic optimization
DETERMINISTIC_PARAMS = STANDARD_PARAMS.copy()

# Parameters for risk-aware optimization
RISK_AWARE_PARAMS = STANDARD_PARAMS.copy()
RISK_AWARE_PARAMS.update({
    'TimeLimit': 2*TimeLimit,    # Longer time limit for risk-aware problems
})

# Parameters for scenario-based optimization
RISK_CONSTRAINED_PARAMS = STANDARD_PARAMS.copy()
RISK_CONSTRAINED_PARAMS.update({
    'TimeLimit': 2*TimeLimit,    # Even longer time limit for scenario-based problems
})


def get_params(optimization_type='standard'):
    """
    Get the appropriate parameter set based on optimization type.

    Args:
        optimization_type (str):
        Type of optimization ('standard', 'deterministic',
                              'risk_aware', or 'risk_constrained')

    Returns:
        dict: Dictionary of Gurobi parameters
    """
    param_sets = {
        'standard': STANDARD_PARAMS,
        'deterministic': DETERMINISTIC_PARAMS,
        'risk_aware': RISK_AWARE_PARAMS,
        'risk_constrained': RISK_CONSTRAINED_PARAMS
    }
    return param_sets.get(optimization_type, STANDARD_PARAMS)
