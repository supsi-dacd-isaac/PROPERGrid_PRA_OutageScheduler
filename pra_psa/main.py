from pandapower.networks import case14
from pra_psa.core.contingency import generate_n1_contingencies
from pra_psa.simulation.subset_simulation import subset_simulation

# simulation/main.py
if __name__ == "__main__":

    net = case14()



    contingencies = generate_n1_contingencies(net)

    def score_fn(net):
        return max(abs(net.res_line.loading_percent.values))

    events, thresholds = subset_simulation(net, contingencies, score_fn)

    print("Rare events:", events)
    print("Thresholds:", thresholds)

