from gurobipy import Model, GRB
from scheduler_monolitic.utils_and_constraints import *
from utils.data_preporcess import *
from utils.data_preporcess import aggregate_hourly_demand as aggregate


class outage_scheduler:
    """  OUTAGE SCHEDULER CLASS  """

    def __init__(self, data):
        # Initialize the model
        self.model = Model("Outage_Scheduler")

        self.names = {
            'T':  [f'step_{t}' for t in range(len(data['nodal_demand']))],  # (List) planning time steps
            'outages': data['outages']['names'],  # (List) outage names
            'lines': [f'line_{l}' for l in range(data['num_branches'])],  # (List) of line names
            'contingencies': data.get('n_minus1_nams', None),  # (List) contingency names
            'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # (List) bus names
            'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()],  # (List) generator names
            'case_name': data['config']['case_name']
        }

        g2b = [f'bus_{g}' for g in data['network'].gen['bus'].values.tolist()]
        _ppc_internal = data['network']._ppc["internal"]
        B_mat = np.real(_ppc_internal['Bf'].A)

        if self.names['contingencies'] is None:  # Generator indices
            self.names['contingencies'] = ([f'n1_{l}' for l in self.names['lines']] +
                                           [f'n1_{g}' for g in self.names['generators']])

        self.params = {
            'p_max':  {gn: (v if v > 0 else 200) for gn, v in zip(self.names['generators'], data['network'].gen['max_p_mw'])},
            'p_min': {gn: v for gn, v in zip(self.names['generators'], data['network'].gen['min_p_mw'])},
            'f_lim': {ln: v for ln, v in zip(self.names['lines'], data['branch_capacity'])},
            'lenT': len(self.names['T']),
            'aggregation_time': data['config']['aggregation_time'],
            'gen2bus': {b: self.names['generators'][idx] for idx, b in enumerate(g2b)},  # (dict): Mapping bus index to generator.
            'net': data['network'],  # pandapower network
            'S': _ppc_internal['Cft'].A.T,  # adjacency matrix
            'B_mat':  B_mat,  #  susceptance matrix
            'B_lines': np.max(np.real(_ppc_internal['Bf'].A), axis=1),  # line-bus susceptance matrix
            'lines2bus': {b: np.argwhere(B_mat[:, b_idx]).flatten() for b_idx, b in enumerate(self.names['buses'])},
            'sign_lines2bus': {b: np.sign(B_mat[np.argwhere(B_mat[:, b_idx]).flatten(), b_idx]) for b_idx, b in enumerate(self.names['buses'])},
            'max_tasks': data['max_number_of_maintenance_tasks'],
            'durations':  {o_nam: data['outages']['expected_duration_steps'][o] for o, o_nam in enumerate(self.names['outages'])},
            'step_cost_outage': {o_nam: data['outages']['cost_per_step'][o] for o, o_nam in enumerate(self.names['outages'])},
        }

        self.weights = {'cost_missmatch': 1e4,
                        'cost_operations': 1e4,
                        'priority': 1e4}

    def build_model(self, nodal_demand):

        self.demand_nominal = nodal_demand

        # Initialize variables
        self._initialize_variables()

        # Define objective function
        self._define_objective()

        # Define constraints
        self._add_all_constraints()

    def _initialize_variables(self):
        """  Add decision variables to the model. """
        # Time period variables (outage start and end times, power generation, flow, curtailment)

        self.X = self.model.addVars(self.names['T'], self.names['outages'], vtype=GRB.BINARY, name="planned_outage_indicator")
        self.start_time = self.model.addVars(self.names['T'], self.names['outages'], vtype=GRB.BINARY, name="end_time")
        self.end_time = self.model.addVars(self.names['T'], self.names['outages'], vtype=GRB.BINARY, name="end_time")

        # Power generation variables for each generator over time
        self.pgen = self.model.addVars(self.names['T'], self.names['generators'],
                                       lb=0, name="power_generation")

        # Curtailment variables (worst-case curtailment at each bus over time)
        self.Delta = self.model.addVars(self.names['T'], self.names['buses'],
                                        lb=0, name="worst_case_curtailment")

        # Transmission line flow and nodal phase variables
        self.flow = self.model.addVars(self.names['T'],  self.names['lines'],
                                       lb=-GRB.INFINITY,   ub=GRB.INFINITY,  name="flow")

        self.theta = self.model.addVars(self.names['T'], self.names['buses'],
                                        lb=-30, ub=30, name="nodal_phase_tb")

        # Contingency variables (for different possible contingencies)
        self.pgen_c = self.model.addVars(self.names['T'], self.names['generators'], self.names['contingencies'],
                                         lb=0, name="power_generation_contingency")

        self.Delta_c = self.model.addVars(self.names['T'], self.names['buses'], self.names['contingencies'],
                                          lb=0, name="worst_case_curtailment_contingency")

        self.flow_c = self.model.addVars(self.names['T'], self.names['lines'], self.names['contingencies'],
                                         lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_contingency")

        self.theta_c = self.model.addVars(self.names['T'], self.names['buses'], self.names['contingencies'],
                                          lb=-30, ub=30, name="nodal_phase_tb_contingency")



        logger.info(f' \n'
                    f'{blue_c}Variables Added:{reset_c}\n'
                    f' - {blue_c}X, start_time, end_time{reset_c}: s Scheduled outage decisions, start-end indicators for the outage task\n'
                    f' - {blue_c}pgen, pgen_c{reset_c}: Generated power normal and N-1 states\n'
                    f' - {blue_c}Delta, Delta_c{reset_c}: Worst-cases demand cut normal and N-1 states\n'
                    f' - {blue_c}flow, flow_c{reset_c}: Power flow in normal and N-1 states\n' 
                    f' - {blue_c}theta, theta_c {reset_c}: Phase angles in normal operation and N-1 states \n'
                    )

    def _define_objective(self):
        """ ----  OBJECTIVE FUNCTION ----"""
        # 1) outages costs
        Objective_fun = quicksum(self.X[t, o] for t in self.names['T'] for o in self.names['outages'])

        # 2) Minimize Load-Generation Missmatch

        Objective_fun -= (self.weights['cost_missmatch'] *
                          quicksum(self.Delta[t, n]
                                   for n in self.names['buses']
                                   for t in self.names['T']))

        Objective_fun -= (self.weights['cost_missmatch'] *
                          quicksum(self.Delta_c[t, n, c]
                                   for n in self.names['buses']
                                   for c in self.names['contingencies']
                                   for t in self.names['T']))

        # 3) Minimize Operational Generation Costs
        Objective_fun -= quicksum(self.pgen[t, g]
                                  for g in self.names['generators']
                                  for t in self.names['T'])

        Objective_fun -= quicksum(self.pgen_c[t, g, c]
                                  for g in self.names['generators']
                                  for c in self.names['contingencies']
                                  for t in self.names['T'])

        self.model.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c} Objective Function Defined:{reset_c} \n'
            f'{green_c} Maximize (total number of planned outages + priorities weight){reset_c} \n'
            f'{green_c} Minimize (Missmatch |demand-generation|...planned + N-1)){reset_c}\n'
            f'{green_c} Minimize (Generation Cost) {reset_c}\n'
        )

    def _add_all_constraints(self):

        logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
        self._add_planned_outages_constraints()

        logger.info('adding power production constraints, planned + unplanned N-1 failures')
        self._add_generators_constraints()

        logger.info('adding line flow constraints, planned + unplanned N-1 failures')
        self._add_line_power_dc_limit_constraints()

        logger.info('adding node balance constraints, planned + unplanned N-1 failures')
        self._add_nodal_power_balance_constraints()

    def _add_planned_outages_constraints(self):
        # 1) Max simultaneous outages
        for t in self.names['T']:
            sum_Xt = quicksum(self.X[t, o] for o in self.names['outages'])
            self.model.addConstr(sum_Xt <= self.params['max_tasks'], name=f"MaxTasks_{t}")

        for o in self.names['outages']:  # 2) Total duration constraint

            self.model.addConstr(quicksum(self.X[t, o] for t in self.names['T']) == self.params['durations'][o], name=f"Duration_{o}")

            # 2.1) Continuity constraints  - - Ensure we don't go out of bounds
            for t_idx, t in enumerate(self.names['T'][:-1]):
                self.model.addConstr(self.start_time[self.names['T'][t_idx + 1], o] >= self.start_time[self.names['T'][t_idx], o], name=f"Cont_start_{o}_{t}")
                self.model.addConstr(self.end_time[self.names['T'][t_idx + 1], o] >= self.end_time[self.names['T'][t_idx], o], name=f"Cont_end_{o}_{t}")
                self.model.addConstr(self.end_time[self.names['T'][t_idx + 1], o] <= self.start_time[self.names['T'][t_idx], o], name=f"end_only_after_start_{o}_{t}")

            # 2.2) Ensure xt = 1 if started but not ended
            for t_idx, t in enumerate(self.names['T']):
                self.model.addConstr(self.start_time[t, o] - self.end_time[t, o] == self.X[t, o], name=f"Cont_start_end_x_{o}_{t}")

    def _add_generators_constraints(self):

        for t_idx, t in tqdm(enumerate(self.names['T']), desc="Adding Constraints",
                             total=self.params['lenT'], ncols=100, colour="green"):

            # Constraints with/without scheduled outage
            for g in self.names['generators']:
                if g in self.names['outages']:
                    self.model.addConstr(self.pgen[t, g] <= self.params['p_max'][g] * (1 - self.X[t, g]), name=f"Gen_{t}_{g}_UP")
                    self.model.addConstr(self.pgen[t, g] >= self.params['p_min'][g] * (1 - self.X[t, g]), name=f"Gen_{t}_{g}_LOW")
                else:
                    self.model.addConstr(self.pgen[t, g] <= self.params['p_max'][g], name=f"Gen_{t}_{g}_UP")
                    self.model.addConstr(self.pgen[t, g] >= self.params['p_min'][g], name=f"Gen_{t}_{g}_LOW")

                # Constraints with/without scheduled outage under N-1 unplanned failure
                for c in self.names['contingencies']:
                    if c[3:] == g:
                        self.model.addConstr(self.pgen_c[t, g, c] <= +0.005, name=f"Gen_{t}_{g}_{c}_UP")
                        self.model.addConstr(self.pgen_c[t, g, c] >= -0.005, name=f"Gen_{t}_{g}_{c}_LOW")
                    else:
                        if g in self.names['outages']:
                            self.model.addConstr(self.pgen_c[t, g, c] <= self.params['p_max'][g] * (1 - self.X[t, g]),
                                                 name=f"Gen_{t}_{g}_{c}_UP")
                            self.model.addConstr(self.pgen_c[t, g, c] >= self.params['p_min'][g] * (1 - self.X[t, g]),
                                                 name=f"Gen_{t}_{g}_{c}_LOW")
                        else:
                            self.model.addConstr(self.pgen_c[t, g, c] <= self.params['p_max'][g], name=f"Gen_{t}_{g}_{c}_UP")
                            self.model.addConstr(self.pgen_c[t, g, c] >= self.params['p_min'][g],
                                                 name=f"Gen_{t}_{g}_{c}_LOW")

    def _add_line_power_dc_limit_constraints_gpt(self):
        # Precompute masks and indices
        B_mat = self.params['B_mat']
        f_lim = self.params['f_lim']
        outages_set = set(self.names['outages'])
        contingencies_set = set(self.names['contingencies'])

        idx_con_ij_list = [B_mat[l_idx, :] != 0 for l_idx in range(len(self.names['lines']))]
        B_ij_list = [B_mat[l_idx, idx_con_ij] for l_idx, idx_con_ij in enumerate(idx_con_ij_list)]

        for t_idx, t in tqdm(enumerate(self.names['T']), desc="Adding Constraints",
                             total=self.params['lenT'], ncols=100, colour="green"):

            # Gather v_phases_t once per timestep
            v_phases_t = np.array([self.theta[t, b] for b in self.names['buses']])

            for l_idx, l in enumerate(self.names['lines']):
                idx_con_ij = idx_con_ij_list[l_idx]
                B_ij = B_ij_list[l_idx]

                # Outage check
                is_outage = l in outages_set
                outage_factor = 1 - self.X[t, l] if is_outage else 1

                flow_lim_l = f_lim[l] * outage_factor
                flow_dc_fact = np.dot(B_ij, v_phases_t[idx_con_ij]) * outage_factor

                # Add constraints for main flow
                self.model.addConstr(self.flow[t, l] <= flow_lim_l, name=f"Flow_{t}_{l}_UP")
                self.model.addConstr(self.flow[t, l] >= -flow_lim_l, name=f"Flow_{t}_{l}_LOW")
                self.model.addConstr(self.flow[t, l] == flow_dc_fact, name=f"Flow_{t}_{l}_DC_eq")

                # Contingency loop
                for c in contingencies_set:
                    # Gather contingency phases
                    v_phases_t_cont = np.array([self.theta_c[t, b, c] for b in self.names['buses']])
                    flow_dc_con = np.dot(B_ij, v_phases_t_cont[idx_con_ij])

                    if c[3:] == l:  # Contingency affecting the line
                        self.model.addConstr(self.flow_c[t, l, c] <= +0.001, name=f"Flow_{t}_{l}_{c}_UP")
                        self.model.addConstr(self.flow_c[t, l, c] >= -0.001, name=f"Flow_{t}_{l}_{c}_LOW")
                        self.model.addConstr(self.flow_c[t, l, c] == flow_dc_con * 0, name=f"Flow_{t}_{l}_{c}_DC_eq")
                    else:
                        # Outage check for contingency
                        self.model.addConstr(self.flow_c[t, l, c] <= flow_lim_l, name=f"Flow_{t}_{l}_{c}_UP")
                        self.model.addConstr(self.flow_c[t, l, c] >= -flow_lim_l, name=f"Flow_{t}_{l}_{c}_LOW")
                        self.model.addConstr(self.flow_c[t, l, c] == flow_dc_fact, name=f"Flow_{t}_{l}_{c}_DC_eq")

    def _add_line_power_dc_limit_constraints(self):


        B_mat = self.params['B_mat']
        f_lim = self.params['f_lim']
        idx_con_ij_list = [B_mat[l_idx, :] != 0 for l_idx in range(len(self.names['lines']))]
        B_ij_list = [B_mat[l_idx, idx_con_ij] for l_idx, idx_con_ij in enumerate(idx_con_ij_list)]

        for t_idx, t in tqdm(enumerate(self.names['T']), desc="Adding Constraints",
                             total=self.params['lenT'], ncols=100, colour="green"):

            v_phases_t = np.array([self.theta[t, b] for b in self.names['buses']])

            for l_idx, l in enumerate(self.names['lines']):
                idx_con_ij = idx_con_ij_list[l_idx]
                B_ij = B_ij_list[l_idx]

                flow_dc = np.dot(B_ij, v_phases_t[idx_con_ij])  # flow_dc = B_l(theta_l[start] - theta_l[end])

                if l in self.names['outages']:
                    cnt1 = self.flow[t, l] <= f_lim[l] * (1 - self.X[t, l])
                    cnt2 = self.flow[t, l] >= -f_lim[l] * (1 - self.X[t, l])
                    cnt_dc = self.flow[t, l] == flow_dc * (1 - self.X[t, l])

                else:
                    cnt1 = self.flow[t, l] <= f_lim[l]
                    cnt2 = self.flow[t, l] >= -f_lim[l]
                    cnt_dc = self.flow[t, l] == flow_dc

                self.model.addConstr(cnt1, name=f"Flow_{t}_{l}_UP")
                self.model.addConstr(cnt2, name=f"Flow_{t}_{l}_LOW")
                self.model.addConstr(cnt_dc, name=f"Flow_{t}_{l}_DC_eq")

                for c in self.names['contingencies']:
                    v_phases_t_cont = np.array([self.theta_c[t, b, c] for b in self.names['buses']])
                    flow_dc_con = np.dot(B_ij, v_phases_t_cont[idx_con_ij])  # flow_dc = B_l(theta_l[start] - theta_l[end])

                    if c[3:] == l:
                        cnt1 = self.flow_c[t, l, c] <= +0.001
                        cnt2 = self.flow_c[t, l, c] >= -0.001
                        cnt_dc = self.flow_c[t, l, c] == flow_dc_con*0

                    else:
                        if l in self.names['outages']:
                            cnt1 = self.flow_c[t, l, c]  <= f_lim[l] * (1 - self.X[t, l])
                            cnt2 = self.flow_c[t, l, c] >= -f_lim[l] * (1 - self.X[t, l])
                            cnt_dc = self.flow_c[t, l, c]  == flow_dc_con * (1 - self.X[t, l])

                        else:
                            cnt1 = self.flow_c[t, l, c]  <= f_lim[l]
                            cnt2 = self.flow_c[t, l, c]  >= -f_lim[l]
                            cnt_dc = self.flow_c[t, l, c] == flow_dc_con

                    self.model.addConstr(cnt1, name=f"Flow_{t}_{l}_{c}_UP")
                    self.model.addConstr(cnt2, name=f"Flow_{t}_{l}_{c}_LOW")
                    self.model.addConstr(cnt_dc, name=f"Flow_{t}_{l}_{c}_DC_eq")

    def _add_nodal_power_balance_constraints(self):

        nodal_balance = self.calculate_nodal_balance(demand=self.demand_nominal)
        for t_idx, t in tqdm(enumerate(self.names['T']), desc="Adding Constraints",
                             total=self.params['lenT'], ncols=100, colour="green"):

            for b_idx, b in enumerate(self.names['buses']):  # Add each constraint individually
                self.model.addConstr(nodal_balance[t_idx, b_idx] <= self.Delta[t, b], name=f'Power_Balance_{t}_{b}_up')
                self.model.addConstr(nodal_balance[t_idx, b_idx] >= -self.Delta[t, b], name=f'Power_Balance_{t}_{b}_low')

        for c in tqdm(self.names['contingencies'], desc="Adding Constraints", total=len(self.names['contingencies']), ncols=100, colour="green"):  # Inflow vector for contingency case
            nodal_balance = self.calculate_nodal_balance(demand=self.demand_nominal, c=c)
            for t_idx, t in enumerate(self.names['T']):  # forall time steps
                for b_idx, b in enumerate(self.names['buses']):
                    self.model.addConstr(nodal_balance[t_idx, b_idx] <= self.Delta_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_up")
                    self.model.addConstr(nodal_balance[t_idx, b_idx] >= -self.Delta_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_low")

    def calculate_nodal_balance(self, demand=None, c=None):
        """
        Calculate inflows, generation, and nodal balance for a given set of parameters.
        Parameters:
            demand (DataFrame): Demand values.
        Returns:
            nodal_balance (ndarray): The calculated nodal balance.
        """

        inflows_mat, generation_all = [], []
        gen2bus = self.params['gen2bus']
        for t in self.names['T']:
            if c is None:
                flows_t = np.array([self.flow[t, l] for l in self.names['lines']])  # Flow values for current time step
            else:
                flows_t = np.array([self.flow_c[t, l, c] for l in self.names['lines']])  # Flow values for current time step

            # Calculate inflows at each bus
            # #todo: check if this is correct
            sum_flows_bus = [np.sum(np.dot(self.params['sign_lines2bus'][b], flows_t[self.params['lines2bus'][b]])) for b_idx, b in enumerate(self.names['buses'])]

            # Calculate generation at each bus
            if c is None:
                generators_t = [self.pgen[t, gen2bus[b]] if b in gen2bus else 0 for b in self.names['buses']]
            else:
                generators_t = [self.pgen_c[t, gen2bus[b], c] if b in gen2bus else 0 for b in self.names['buses']]

            inflows_mat.append(sum_flows_bus)
            generation_all.append(generators_t)

        # Convert lists to NumPy arrays and remove any extra dimensions
        inflows_mat = np.squeeze(np.array(inflows_mat))
        generation_all = np.squeeze(np.array(generation_all))

        # Calculate nodal balance
        nodal_balance = demand.values - inflows_mat - generation_all

        return nodal_balance

    def solve(self, save_res_name=None):

        if save_res_name is None:
            save_res_name = "optimal_solution" + self.names['case_name'] + '_' + self.params['aggregation_time'] + ".json"
            save_res_name = '../data/schedule_results/deterministic_optimizer/' + save_res_name

            # ----  SOLVE the M
            self.model.setParam('MIPGap', 0.1)  # Acceptable optimality gap
            self.model.setParam('Heuristics', 0.5)  # Emphasize heuristics
            self.model.setParam('Cuts', 2)  # Allow Gurobi to generate more cuts
            self.model.setParam('Presolve', 2)  # Enable aggressive pre-solve
            self.model.setParam('Threads', 8)  # Use 8 threads for parallel computation
            self.model.setParam('TimeLimit', 1200)  # Set a time limit

            self.model.update()
            self.model.optimize()

            """self.model.setParam('DualReductions', 0)
            self.model.computeIIS()
            self.model.write('iis.ilp') """

            # Check optimization status
            if self.model.status == GRB.OPTIMAL:
                logger.info(f"{blue_c} Optimal solution found! :-) :-):-){reset_c}")
                solution = {v.VarName: v.X for v in self.model.getVars()}  # Retrieve and save variable values
                with open(save_res_name, "w") as f:  # Save the solution to a file or database
                    json.dump(solution, f)

                (SOLUTION, X_OP, WC_cut_norm, WC_cut_con, Pgen, Flows, Prod) \
                    = post_process_results(save_res_name, self.names, T=self.names['T'])

                visualize_results(X_OP, Pgen, Flows, WC_cut_norm, WC_cut_con, self.names)  # plot

                return SOLUTION, X_OP, WC_cut_norm, WC_cut_con, Pgen

            elif self.model.status == GRB.INFEASIBLE:
                logger.warning(f"{red_c} Model is infeasible! :-(:-(:-({reset_c}")
                # Run IIS to find conflicting constraints
                self.model.computeIIS()
                self.model.write("model.ilp")  # Write IIS to a file for inspection
                print("Conflicting constraints are:")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(c.ConstrName)
                return None
            elif self.model.status == GRB.UNBOUNDED:
                logger.warning("Model is unbounded.")  # Save unbounded M status to log or file
                with open("M_status.txt", "a") as f:
                    f.write("Model is unbounded.\n")
                return None
            else:
                logger.info(f"Optimization was stopped with status: {self.model.status}")
                return None


if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """
    # Load data
    network, hourly_demand, config = data_loader('../config/wp3_optim/IEEE24_scheduler.json')
    aggregation_step = config['aggregation_time']  # 'W', 'D', 'H
    cost_per_days = [1000, 2000, 1000, 1000, 2000, 1000, 2000, 5000, 10]
    expected_duration_days = [25, 14, 55, 14, 30, 30, 30, 60, 20]
    daily_demand = aggregate(hourly_demand * 0.5, aggregation_step=aggregation_step)
    if aggregation_step == 'H':
        daily_demand = daily_demand.iloc[:4000, :]  # limit the number of steps
        cost_per_step = [c_day / 24 for c_day in cost_per_days]
        expected_duration_steps = [d * 24 for d in expected_duration_days]
    elif aggregation_step == 'W':
        cost_per_step = [c_day * 7 for c_day in cost_per_days]
        expected_duration_steps = [int(d / 7) for d in expected_duration_days]
    else:
        cost_per_step = cost_per_days
        expected_duration_steps = expected_duration_days

    # example scheduled outage information
    outages = {'indices': [0, 1, 2, 3, 5, 8, 1, 2, 8],
               'names': ['line_0', 'line_1', 'line_2', 'line_3', 'line_5', 'line_8', 'gen_1', 'gen_2', 'gen_8'],
               'type': ['line', 'line', 'line', 'line', 'line', 'line', 'generator', 'generator', 'generator'],
               'expected_duration_steps': expected_duration_steps,  # step_are_in_days for now
               'cost_per_step': cost_per_step,
               'priorities': [1, 2, 1, 3, 2, 1, 1, 3, 2]}

    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in
                                                                                                  network.trafo]

    data = {'max_number_of_maintenance_tasks': 2,
            'nodal_demand': daily_demand,
            'outages': outages,
            'config': config,
            'network': network,
            'num_buses': num_buses,
            'num_branches': num_branches,
            'ref_buses': ['bus_12'],
            'branch_capacity': branch_capacity,
            'n_minus1_names': [f'n1_{l}' for l in [f'line_{k}' for k in range(30)]]}

    # dic_res = SCOS_no_phases(data)
    SCHEDULER = outage_scheduler(data)
    SCHEDULER.build_model(nodal_demand=daily_demand)

    SCHEDULER.solve()

