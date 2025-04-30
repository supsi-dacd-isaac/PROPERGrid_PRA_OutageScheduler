import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.gurobi_SCOS import *

from optimizers.vis_scos_res import plot_comparison_CVAR_DET_SCOS

from optimizers.pre_post_processing import (hash_simulation_params,  simulation_exists,  load_metadata, save_metadata)


def run_optimizer(data, optimizer_type='deterministic', cvar_limit=50):
    """
    Run the specified optimizer on the given data.
    Args:
        data (dict): Dictionary containing the problem data
        optimizer_type (str): Type of optimizer to use ('deterministic' or 'risk_aware')
    Returns:
        dict: Solution containing the optimal schedule and objective value
    """
    # Create results directory if it doesn't exist
    results_dir = Path(f"results/{optimizer_type}_SCOP_optimizer")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Generate simulation parameters and hash
    params = {
        'optimizer_type': optimizer_type,
        'VoLL': data['VoLL'],
        'tail_prob_constraint': data['tail_prob_constraint'],
        'n_samples': data['n_samples'],
        'use_DC_PF': data['use_DC_PF'],
        'aggregation_step': data['config']['aggregation_time'],
        'scaling_factor': data['config']['load_scaling_factor'],
        'max_number_of_maintenance_tasks': data['max_number_of_maintenance_tasks']
    }

    # Check if simulation already exists
    sim_hash = hash_simulation_params(params)
    sim_filename = f"sim_{sim_hash}"

    if simulation_exists(results_dir, sim_filename):
        logging.info(f"Loading existing simulation results for {sim_filename}")
        return load_metadata(results_dir, sim_filename)

    # Run the appropriate optimizer
    if optimizer_type == 'deterministic':
        results_dictionary, solution, Objective_optimal_val = run_SCOS_deterministic(data)
    elif optimizer_type == 'risk_aware':
        results_dictionary, solution, Objective_optimal_val = run_SCOS_cvar(data)
    elif optimizer_type == 'risk_constrained':
        results_dictionary, solution, Objective_optimal_val = run_SCOS_cvar(data, cvar_limit=cvar_limit)
    else:
        logging.error(f"Invalid optimizer type: {optimizer_type}")
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Check if optimization was successful
    if results_dictionary is None or solution is None or Objective_optimal_val is None:
        logging.error(f"Optimization failed for {optimizer_type}")
        return None, None, None, None

    # Save results
    save_metadata(results_dictionary, results_dir, sim_filename, solution=solution, objective=Objective_optimal_val, params=params)
    return results_dictionary, solution, Objective_optimal_val, params


# Example How to run
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """

    # Load system data and historical nodal demand data
    config_path  = 'config/conf_IEEE24.json'
    DATA = load_data_from_conf_grid_case(conf_path=config_path)
    names = DATA['names']

    params = {
        'config_path': config_path,
        'VoLL': DATA['VoLL'],
        'tail_prob_constraint': DATA['tail_prob_constraint'],
        'n_samples': DATA['n_samples'],
        'use_DC_PF': DATA['use_DC_PF'],
        'aggregation_step': DATA['config']['aggregation_time'],
        'scaling_factor': DATA['config']['load_scaling_factor'],
        'max_number_of_maintenance_tasks': DATA['max_number_of_maintenance_tasks']
    }

    # Run deterministic optimization
    print("Running deterministic optimization...")
    det_results = run_optimizer(DATA, optimizer_type='deterministic')

    # Run risk-aware optimization
    print("Running risk-aware optimization...")
    risk_results = run_optimizer(DATA, optimizer_type='risk_aware')


    # Run risk-aware optimization
    print("Running risk-aware optimization...")
    risk_results = run_optimizer(DATA, optimizer_type='risk_constrained')


    # Compare results
    plot_comparison_CVAR_DET_SCOS(risk_results[0], det_results[0], DATA)

    print("Objective (DET):", det_results[2])
    print("Objective (RISK):", risk_results[2])
