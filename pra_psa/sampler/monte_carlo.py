
import pandapower as pp


def run_MC(net,  load_scenarios, base_load =None, outage_set = None):
    if base_load is not None:
        net.load['p_mw'] = base_load

    if outage_set is not None:

            for type, idx in outage_set:
                try:
                    if type == 'line':
                        net.line.at[idx, 'in_service'] = False
                    if type == 'gen':
                        net.gen.at[idx, 'in_service'] = False
                    if type == 'trafo':
                        net.trafo.at[idx, 'in_service'] = False
                except:
                    print(f'type {type} not recognized')

    pp.runopp(net,calculate_voltage_angles=True, init='flat', delta=1e-8, verbose=False)
    net.gen['p_mw']  =      net.res_gen['p_mw']
    net.gen['vm_pu'] =      net.res_gen['vm_pu']
    #net.gen['q_mvar'] =     net.res_gen['q_mvar']
    net.gen['controllable'] = False

    results = []
    for i, p_mw in enumerate(load_scenarios):
        NET = net.deepcopy()

        try:
            NET.load['p_mw'] = p_mw
            pp.runpp(net,calculate_voltage_angles=True, init='flat', delta=1e-8, verbose=False)
            converged = True
            results.append({'scenario':i, 'converged':converged,
                            'res_line': NET.res_line.copy(), 'res_bus': NET.res_bus.copy(), 'res_gen':NET.res_gen.copy(),
                            'loading_percent': NET.res_line['loading_percent'], 'vm_pu': NET.res_gen['vm_pu'].values,
                            'ext_grid_p_mw': NET.res_ext_grid['p_mw'].values,
                            'max_vm_pu':NET.res_gen['vm_pu'].max(), 'min_vm_pu': NET.res_gen['vm_pu'].min(), 'max_loading': NET.res_line['loading_percent'].max() })
        except pp.powerflow.LoadflowNotConverged:
            converged = False
            results.append({'scenario':i, 'converged':converged,
                             'res_line': None, 'res_bus': None, 'res_gen':None,
                            'loading_percent':None, 'vm_pu': None, 'ext_grid_p_mw': None,
                            'max_vm_pu':None, 'min_vm_pu':None, 'max_loading': None })

