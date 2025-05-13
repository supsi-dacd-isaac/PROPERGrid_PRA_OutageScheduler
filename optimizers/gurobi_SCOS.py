import numpy as np
from gurobipy import GurobiError
from optimizers.build_problem_gurobi import *
from optimizers.pre_post_processing import *
from optimizers.vis_scos_res import visualize_results
import gurobipy as gp
from gurobipy import GRB
from optimizers.gurobi_params import get_params
from pra_psa.utils.utils import *
from pra_psa.probabilistic_models.demand_sampler import demand_sampler
from optimizers.vis_scos_res import plot_risk_pdf_cdf
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def initialize_model(problem_type='deterministic'):
    """ Build a Gurobi optimization problem based on the specified type.
    Args:
        problem_type (str): Type of problem to build ('deterministic' or 'risk_aware')
    Returns:
        tuple: (model, variables) where model is the Gurobi model and variables is a dict of variables
    """
    params = get_params(problem_type)   # Get parameters based on problem type
    model = gp.Model(f"SCOS_{problem_type}")   # Create model
    for param, value in params.items():  # Set parameters
        model.setParam(param, value)
    return model


def logger_print_info(problem_type: str = '',
                      msg: str = ''):
    """ Print information to the logger.
    Args:
        msg (str): Message to log
    """
    if problem_type == 'risk' and msg == 'objective':
        msg = (f'{blue_c} Objective Function Defined: ∑ Outage_NumberCost_Priority'
               f' - ∑ LossOfLoad(N-1 and Normal) - '
               f' - ∑ CostOperations(Pgen) '
               f' - ∑ CVaR(LossOfLoad){reset_c}\n' )

    elif problem_type == 'deterministic' and msg == 'objective':
        msg = (f'{blue_c} Objective Function Defined: ∑ Outage_NumberCost_Priority '
               f' - ∑ LossOfLoad(N-1 and Normal) - '
               f' - ∑ CostOperations(Pgen) {reset_c}\n')
    elif msg == 'variables':
        msg = ( f' \n'
                    f'{blue_c}Variables Added:{reset_c}\n'
                    f' - {blue_c}xt, sxt, ext{reset_c}: s Scheduled outage decisions, start-end indicators for the outage task\n'
                    f' - {blue_c}pgen, pgen_c{reset_c}: Generated power planned and N-1 states\n'
                    f' - {blue_c}d_cut, d_cut_c{reset_c}: loss of load planned and N-1 states\n'
                    f' - {blue_c}f, f_c{reset_c}: Power flow in planned and N-1 states\n')

    logger.info(msg)


def run_SCOS_deterministic(data):
    """  Deterministic security-constrained outage planner using Gurobi.
    Args:
        data (dict): Dictionary containing system data and parameters
    Returns:
        tuple: (results dictionary, solution object, objective value)
    """

    try:
        # start with model definition
        model = initialize_model('deterministic')
        names, T, VoLL = data['names'], data['T'], data['VoLL']

        # Initialize variables
        model, VARIABLES = initialize_variables(model=model, data=data)
        logger_print_info(msg='variables')

        # Define objective function
        model, objective_fun, objective_terms = define_objective_fun(model=model,
                                                                     data=data,
                                                                     variables=VARIABLES)
        logger_print_info('deterministic', msg='objective')

        # Add constraints
        model = add_deterministic_constraints(model, data, VARIABLES)

        # Solve the model
        model.update()
        model.optimize()
        solution = get_gurobi_solution(model)

        if solution is not None:
            results_dictionary = post_process_results(solution, names, T=T)
            visualize_results(results_dictionary, names)

            total_objective_val = model.objVal
            objective_val = model.objVal

            # Prepare dictionary with all terms
            obj_val_composed = {
                'Total Objective': total_objective_val,
                'Det Objective': objective_val,
                'PM cost term': objective_terms['PM'].getValue(),
                'VoLL Normal': objective_terms['VoLL'].getValue(),
                'Generation Cost Normal': objective_terms['Generation Cost'].getValue(),
                'VoLL Contingency': objective_terms['VoLL Contingency'].getValue(),
                'Generation Cost Contingency': objective_terms['Generation Cost Contingency'].getValue(),
                'CVaR Risk Contribution': []
            }
            return results_dictionary, solution, obj_val_composed
        else:
            return None, solution, None

    except GurobiError as e:
        logger.error(f"Error code {e.errno}: {e}")
    except Exception as e:
        logger.error(f"Exception during optimization: {e}")





def run_SCOS_cvar(data, cvar_limit=None, percentile=None):
    """
    Solve the SCOS problem with CVaR as a constraint using Gurobi.
    Args:
        data (dict): Dictionary containing the problem data
        cvar_limit (float, optional): Upper limit for CVaR constraint
        percentile (float): Percentile for CVaR calculation (default: 0.95)
    Returns:
        tuple: (results dictionary, solution object, objective value)
    """

    # start with model definition
    model = initialize_model('risk_aware')
    names, T, VoLL = data['names'], data['T'], data['VoLL']

    # Initialize variables
    model, VARIABLES = initialize_variables(model=model, data=data)
    logger_print_info(msg='variables')

    # Define objective function
    model, objective_fun, objective_terms = define_objective_fun(model=model,
                                                                 data=data,
                                                                 variables=VARIABLES)

    logger_print_info('risk', msg='objective')

    if percentile is not None:
        if not (0.0 <= percentile <= 1.0):
            raise ValueError("Percentile must be between 0 and 1")
        tail_prob = 1 - percentile
    else:
        tail_prob = data.get('tail_prob_constraint', 0.0) or 0.0

    N = len(names['demand_scenarios'])
    prob_scen = {s: 1 / N for s in names['demand_scenarios']}

    # Add constraints
    model = add_deterministic_constraints(model, data, VARIABLES)

    # Scenario balance → shortage
    logger.info('Generating demand scenarios')
    #    Precompute a (Sn × TB) NumPy matrix of balances
    nodal_demand_samples = demand_sampler(data['nodal_demand'],
                                          n_samples=len(names['demand_scenarios']),
                                          random_seed=42)

    logger.info('Adding scenario variables, CVAR objective and/or constraints ')
    model, CVaR = add_CVaR(model=model, data=data, Variables=VARIABLES,
                           d_samples=nodal_demand_samples, prob_scen=prob_scen,
                           tail_prob=tail_prob, cvar_limit=cvar_limit,
                           objective_fun=objective_fun)

    model.update()
    model.optimize()

    solution = get_gurobi_solution(model)

    if solution is not None:
        results_dictionary = post_process_results(solution, names, T=T)
        visualize_results(results_dictionary, names)
        plot_risk_pdf_cdf(results_dictionary['CVARisk'])

        total_objective_val = model.objVal
        objective_val = model.objVal - sum(
            CVaR[t, b].x for t in T for b in names['buses'])

        obj_val_composed = {
            'Total Objective': total_objective_val,
            'Det Objective': objective_val,
            'PM cost term': objective_terms['PM'].getValue(),
            'VoLL Normal': objective_terms['VoLL'].getValue(),
            'Generation Cost Normal': objective_terms['Generation Cost'].getValue(),
            'VoLL Contingency': objective_terms['VoLL Contingency'].getValue(),
            'Generation Cost Contingency': objective_terms['Generation Cost Contingency'].getValue(),
            'CVaR Risk Contribution': sum(CVaR[t, b].x for t in T for b in names['buses'])
        }
        return results_dictionary, solution, obj_val_composed
    else:
        return None, solution, None



