from scheduler.dataprocess import get_solution_dic, prepare_data
from scheduler.variables import add_variables
from scheduler.constraints import add_planned_outages_constraints, add_all_generator_constraints, add_line_power_limit_constraints, add_nodal_power_balance_constraints
from scheduler.objective import objective_function
from visualization.visualize_schedule import visualize_results
from gurobipy import Model, GRB


def main():

    DATA = prepare_data(conf_path='./config/conf_IEEE24_scheduler_v2.json')
    model_name = 'deterministic_optimizer'
    model = Model(f"Transmission_Outage_Scheduling_{model_name}")
    model, VARIABLES = add_variables(model=model, data=DATA, optim_type='deterministic')
    Objectives = objective_function(data=DATA, variables=VARIABLES)
    model.setObjective(Objectives['Total'], GRB.MAXIMIZE)
    model = add_planned_outages_constraints(model=model, variables=VARIABLES, data=DATA)
    model = add_all_generator_constraints(model=model, variables=VARIABLES, data=DATA)
    model = add_line_power_limit_constraints(model=model, variables=VARIABLES, data=DATA)
    model = add_nodal_power_balance_constraints(model=model, variables=VARIABLES, data=DATA)

    # model = add_nodal_power_balance_constraints_sparse(model=model, variables=VARIABLES, data=DATA)
    # @title SOLVE the Model
    model.setParam('MIPGap', 0.05)  # Acceptable optimality gap
    model.setParam('Heuristics', 0.5)  # Emphasize heuristics
    model.setParam('Cuts', 2)  # Allow Gurobi to generate more cuts
    model.setParam('Presolve', 2)  # Enable aggressive pre-solve
    model.setParam('Threads', 8)  # Use 8 threads for parallel computation
    model.setParam('TimeLimit', 3600)  # Set a one-hour time limit
    model.update()
    model.optimize()

    res_dic_vars, res_obj = get_solution_dic(model=model, data=DATA)
    visualize_results(results_dictionary=res_dic_vars, names=DATA['names'])

if __name__ == "__main__":

    main()