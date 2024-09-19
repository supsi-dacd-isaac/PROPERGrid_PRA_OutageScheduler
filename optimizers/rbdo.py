### implementation of risk-based/risk-constrained and reliability-based/reliability(chance)-constrained optimization
import gurobipy as gp
from gurobipy import GRB
from typing import Callable, List, Optional, Tuple


def hard_constrained_optimization(g_reliability: Callable,  n_var:int,
                                  delta: List,  f_cost: Callable,
                                  lb: float = -float('inf'),
                                  ub: float = float('inf'),
                                  convex: bool = True):
    """  Hard-constrained optimization problem with general objective and constraints.
    Args:
    - g_reliability: Callable = Reliability performance function.... g(omega, s[i]) <= 0 (defines the safe domain)
    - n_var: Number of decision variables.
    - delta: Dataset, where each s[i] is a random sample.
    - f_cost: Callable = cost function f(omega)
    Returns: - Optimal omega solution. """
    model = gp.Model("hard_constrained_optimization")
    x = model.addVars(n_var, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="decision variables")
    model.setObjective(f_cost(x), GRB.MINIMIZE)  # Set objective function
    for i, s in enumerate(delta):
        model.addConstr(g_reliability(x, s) <= 0, name=f"constraint_{i}")
    if not convex:
        model.setParam('NonConvex', 2)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return [x[i].X for i in range(n_var)]
    else:
        return None


def chance_constrained_optimization(g_reliability: Callable,
                                    n_var: int,
                                    delta: List,
                                    f_cost: Callable,
                                    lb: float = -float('inf'),
                                    ub: float = float('inf'),
                                    alpha: float = 0.05,
                                    convex: bool = True) -> Tuple[List[float], List[float]]:
    """
    Chance-constrained optimization problem with binary variables and general objective/constraints.

    Args:
    - g_reliability: Constraint function g(omega, s[i]) (callable).
    - n_var: Number of decision variables.
    - delta: Dataset, where each s[i] introduces a constraint.
    - f_cost: Objective function of omega (callable).
    - lb: Lower bound for decision variables.
    - ub: Upper bound for decision variables.
    - alpha: Tolerance for constraint violations (fraction of N).
    - convex: Boolean flag for convexity of the problem.

    Returns:
    - Optimal omega solution and binary variables z.
    """
    # Initialize parameters
    N = len(delta)  # Sample size
    Nf_max = int(N * alpha)  # Maximum number of violated constraints allowed
    M = 1000  # Big-M value for constraint relaxation
    # Create a new Gurobi model
    model = gp.Model("chance_constrained_optimization")
    # Decision variables
    x = model.addVars(n_var, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="decision_variables")
    z = model.addVars(N, vtype=GRB.BINARY, name="binary_slack_z")  # Binary variables for soft constraints
    # Set the objective function
    model.setObjective(f_cost(x), GRB.MINIMIZE)
    # Add soft constraints with relaxation using big-M
    for i, s in enumerate(delta):
        model.addConstr(g_reliability(x, s) <= M * z[i], name=f"soft_reliability_constraint_{i}")
    # Add the chance constraint: sum of binary variables should be less than or equal to alpha * N
    model.addConstr(gp.quicksum(z[i] for i in range(N)) <= Nf_max, name="chance_constraint")
    # Set non-convex parameter if needed
    if not convex:
        model.setParam('NonConvex', 2)
    # relax and optimize
    relaxed_model = model.relax()  # Relax the model to an LP model
    relaxed_model.optimize()  # Optimize the relaxed LP model
    model.optimize()  # Optimize the original model
    # Return the optimal solution for omega and z
    if relaxed_model.status == GRB.OPTIMAL:
        decision_variable = [x[i].X for i in range(n_var)]
        z_values = [z[i].X for i in range(N)]
        return decision_variable, z_values
    else:
        return None, None


def risk_constrained_optimization(f_cost: Callable,
                                  g_reliability: Callable,
                                  n_var: int,
                                  delta: List,
                                  lb: float = -float('inf'),
                                  ub: float = float('inf'),
                                  alpha: float = 0.05,
                                  risk_weight: float = 0.2,
                                  convex: bool = True) -> Tuple[float, List[float], List[float]]:
    """
    Risk-constrained optimization problem with slack variables.

    Args:
    - f_omega: Objective function of omega (callable).
    - g_reliability: Constraint function g(omega, s[i]) (callable).
    - delta: Dataset, where each s[i] introduces a constraint.
    - alpha: Tolerance for constraint violations (fraction of N).
    - risk_weight: Weight for penalizing slack variables in the objective function.
    - convex: Boolean flag for convexity of the problem.

    Returns:
    - Optimal omega, slack variables zeta, and binary variables z.
    """
    N = len(delta)  # Sample size
    Nf_max = int(N * alpha)  # Maximum number of constraint violations allowed

    # Weights for the objective function
    weights = {
        'risk': max(min(1.0, risk_weight), 0.0),
        'cost': 1 - max(min(1.0, risk_weight), 0.0)
    }

    # Create a new Gurobi model
    model = gp.Model("risk_constrained_optimization")

    # Decision variables
    design = model.addVars(n_var, lb=lb, ub=ub,
                           vtype=GRB.CONTINUOUS, name="decision_variables")

    z = model.addVars(N,
                      vtype=GRB.BINARY, name="z")  # Binary variables for constraint relaxation

    zeta = model.addVars(N,
                         vtype=GRB.CONTINUOUS, lb=0, name="zeta")  # Slack variables for constraint violations

    # Set the objective function
    model.setObjective(weights['cost'] * f_cost(design) + weights['risk'] * gp.quicksum(zeta[i] for i in range(N)), GRB.MINIMIZE)

    # Add constraints: g(omega, s[i]) <= zeta[i] * z[i]
    for i, s in enumerate(delta):
        model.addConstr(g_reliability(design, s) <= zeta[i] * z[i],
                        name=f"soft_risk_constraint_{i}")

    # Add the chance constraint: sum(z) <= N * alpha
    model.addConstr(gp.quicksum(z[i] for i in range(N)) <= Nf_max,
                    name="chance_constraint")

    # Set non-convex parameter if needed
    if not convex:
        model.setParam('NonConvex', 2)

    # Optimize the model
    model_relaxed = model.relax()
    model_relaxed.optimize()
    model.optimize()  # Optimize the original model

    # Return the optimal solution for omega, z, and zeta
    if model_relaxed.status == GRB.OPTIMAL:
        decision_variable = [design[i].X for i in range(n_var)]
        z_values = [z[i].X for i in range(N)]
        zeta_values = [zeta[i].X for i in range(N)]
        return decision_variable, z_values, zeta_values
    else:
        return None, None, None
