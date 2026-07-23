from utils import data_loader
from utils import aggregate_hourly_demand

import matplotlib.pyplot as plt
import pandapower.plotting as plot_net

if __name__ == '__main__':
    # Example: load demand data, network model, and configuration files
    conf_path = '../config/conf_IEEE118.json'
    net_data118, df_loads118, conf118 = data_loader(conf_path)

    conf_path = '../config/conf_IEEE24_scheduler.json'
    net_data_24, df_loads_24, conf24 = data_loader(conf_path)

    df_loads_daily = aggregate_hourly_demand(df_loads_24)

    # Plot IEEE 118-bus network and IEEE 24-bus network on the second panel
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_net.simple_plot(net_data118, ax=ax[0], plot_loads=True, load_size=2.0, show_plot=False)
    ax[0].set_title("IEEE 118-bus Network")
    plot_net.simple_plot(net_data_24, ax=ax[1], plot_loads=True, load_size=2.0, show_plot=False)
    ax[1].set_title("IEEE 24-bus Network")
    plt.tight_layout()
    plt.show()


    # Plot demand data for IEEE 118-bus and IEEE 24-bus
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    df_loads118.plot(ax=ax[0], title="Hourly Demand - IEEE 118-bus System")
    ax[0].set_title("Hourly Demand - IEEE 118-bus System")
    df_loads_24.plot(ax=ax[1], title="Hourly Demand - IEEE 24-bus System")
    ax[1].set_title("Hourly Demand - IEEE 24-bus System")
    plt.tight_layout()
    plt.show()
