
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import ceil
sns.set_theme(style="whitegrid", palette="colorblind")


def visualize_results(results_dictionary, names):
    """visualize result of the SCOS problem"""

    o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']

    # Unpack results
    X_OutageSchedule = results_dictionary["X_OutageSchedule"]
    PowerGenerated = results_dictionary["PowerGenerated"]
    Line_Flows = results_dictionary["Line_Flows"]
    WC_CURTAIL = results_dictionary["WC_CURTAIL"].T
    WC_CURTAIL_CON = results_dictionary["WC_CURTAIL_CON"]

    # clean up the headers to enhance visualization
    def clan_time_name(c):
        return int(c.replace("step_", "")) if isinstance(c, str) and "step_" in c else c

    PowerGenerated.columns = [clan_time_name(c) for c in PowerGenerated.columns]
    Line_Flows.columns = [clan_time_name(c) for c in Line_Flows.columns]
    WC_CURTAIL_CON.columns = [clan_time_name(c) for c in WC_CURTAIL_CON.columns]
    X_OutageSchedule.columns = [clan_time_name(c) for c in X_OutageSchedule.columns]

    # 🔶 1. Outage Schedule Heatmap
    plt.figure(figsize=(12, len(o_nam) * 0.5 + 2))
    sns.heatmap(X_OutageSchedule, cmap="Blues", cbar=False)
    plt.title("Outage Schedule Heatmap")
    plt.xlabel("Time")
    plt.ylabel("Outage")
    plt.tight_layout()
    plt.show()

    # 🔶 2. Number of PM Tasks Over Time
    plt.figure(figsize=(10, 3))
    X_OutageSchedule.sum().plot(marker='o', linestyle='-')
    plt.title("Number of Planned Maintenance Tasks Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("# Outages")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 🔶 3. Curtailment under Contingencies
    WC_CURTAIL_CON_T = WC_CURTAIL_CON.loc[(WC_CURTAIL_CON != 0).any(axis=1)]
    if not WC_CURTAIL_CON_T.empty:
        WC_CURTAIL_CON_T.T.plot(figsize=(12, 5), linewidth=1.2)
        plt.title("Curtailment Under N-1 Contingencies")
        plt.xlabel("Time Step")
        plt.ylabel("Load Shed [MW]")
        plt.legend(title="Contingency", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("No N-1 curtailment to visualize.")


    # 🔶 4. Total Duration of Each Outage
    plt.figure(figsize=(10, 4))
    sns.barplot(x=X_OutageSchedule.T.sum().index, y=X_OutageSchedule.T.sum().values, color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Total Duration of Each Outage")
    plt.xlabel("Outage ID")
    plt.ylabel("Duration [steps]")
    plt.tight_layout()
    plt.show()

    # 🔶 5. Total Curtailment over time
    WC_CURTAIL.plot(marker='x', linestyle='-', color='red')
    plt.title("Total Load Curtailment Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Curtailment [MW]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 🔶 6. N-1 Contingency Curtailments (Filtered)
    WC_CURTAIL_CON_T = WC_CURTAIL_CON.loc[(WC_CURTAIL_CON != 0).any(axis=1)]
    if not WC_CURTAIL_CON_T.empty:
        WC_CURTAIL_CON_T.T.plot(figsize=(12, 5), linewidth=1.2)
        plt.title("Curtailment Under N-1 Contingencies")
        plt.xlabel("Time Step")
        plt.ylabel("Load Shed [MW]")
        plt.legend(title="Contingency", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No N-1 curtailment to visualize.")

    # 🔶 7. Power Generation Stats
    power_sum = PowerGenerated.sum()
    plt.figure(figsize=(10, 3))
    power_sum.plot(kind='line', marker='o', color='black')
    plt.title("Total Power Generation Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("MW")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Transpose to have time on index
    PG_time_indexed = PowerGenerated.T
    # Infer time step (rows = days or hours)
    n_steps = PG_time_indexed.shape[0]
    if n_steps >= 350 and n_steps <= 400:
        freq = 'D'  # daily
    elif n_steps >= 8000:
        freq = 'H'  # hourly
    else:
        freq = 'D'  # fallback default
    # Define the correct start date based on your data.
    # If your dataset starts in 2023, we adjust the starting point accordingly.
    start_date = '2023-01-01'
    # Create DatetimeIndex based on data shape and inferred frequency
    PG_time_indexed.index = pd.date_range(start=start_date, periods=n_steps, freq=freq)
    total_generation = PG_time_indexed.sum(axis=1)
    weekly_generation = total_generation.resample('W').mean()
    weekly_variability = total_generation.resample('W').std()

    plt.figure(figsize=(10, 4))
    weekly_generation[1:-1].plot(marker='o', color='darkgreen', label='Weekly Generation')
    plt.fill_between(weekly_generation.index[1:-1],
                     weekly_generation[1:-1] - weekly_variability[1:-1],
                     weekly_generation[1:-1] + weekly_variability[1:-1],
                     color='gray', alpha=0.2, label='Variability (Std. Dev.)')
    plt.title("Weekly Power Generation Trend with Variability")
    plt.xlabel("Week")
    plt.ylabel("Total MW (per week)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 🔶 8. Line Flow Stats
    # Ensure the columns are converted to datetime if they're not already
    Line_Flows.columns = pd.date_range(start=start_date, periods=n_steps, freq=freq)
    # Check if the conversion was successful
    if Line_Flows.columns.isnull().any():
        print("Some columns could not be converted to datetime.")
    # Resample to weekly frequency (mean and std dev for variability)
    weekly_line_flows = Line_Flows.T.resample('W').mean()  # Resample along the columns (time dimension)
    weekly_line_flows_std = Line_Flows.T.resample('W').std()  # Standard deviation for variability
    # Limit to the first 20 lines
    n_lines_to_plot = 20
    weekly_line_flows = weekly_line_flows.iloc[:, :n_lines_to_plot]
    weekly_line_flows_std = weekly_line_flows_std.iloc[:, :n_lines_to_plot]
    n_lines = weekly_line_flows.shape[1]  # Total number of lines (rows in the transposed data)
    n_cols = 5
    n_rows = (n_lines + n_cols - 1) // n_cols  # Calculate rows needed to fit all lines
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axes = axes.flatten()
    # Plot each line's weekly data
    for i in range(n_lines):
        ax = axes[i]
        ax.plot(weekly_line_flows.index, weekly_line_flows.iloc[:, i], label=f'Line {i + 1}', color='tab:blue')
        ax.fill_between(weekly_line_flows.index,
                        weekly_line_flows.iloc[:, i] - weekly_line_flows_std.iloc[:, i],
                        weekly_line_flows.iloc[:, i] + weekly_line_flows_std.iloc[:, i],
                        alpha=0.2, label="±1 std dev", color='tab:blue')
        ax.set_title(f'Line {i + 1}')
        ax.set_xlabel('Weeks')
        # Set the x-ticks to only show the first and last week
        xticks = [weekly_line_flows.index[0], weekly_line_flows.index[-1]]  # First and last week
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"Week 1",  f"Week 52"], rotation=45, ha="right")
        ax.set_ylabel('MW')
        ax.grid(True)
    for i in range(n_lines, len(axes)):
        axes[i].axis('off')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


    # 🔶 9. Generator Distribution by Day
    daily_avg_gen = PowerGenerated.T.groupby(PowerGenerated.columns).mean()
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=daily_avg_gen)
    plt.title("Generation Distribution Across Generators")
    plt.ylabel("MW")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 🔶 10. Line Flow Distribution
    daily_avg_flow = Line_Flows.T.groupby(Line_Flows.columns).mean()
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=daily_avg_flow)
    plt.title("Line Flow Distribution Across Lines")
    plt.ylabel("MW")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_risk_pdf_cdf(CVARisk):

    # Aggregate CVaR across all buses for each time step
    cvar_over_time = CVARisk.sum(axis=0)  # Sum of CVaR for all buses at each step

    # 🔶 1. cvar over time
    plt.figure(figsize=(10, 6))
    plt.plot(cvar_over_time, marker='o', linestyle='-', color='red')
    plt.title("CVaR Evolution Over Time (Global Aggregation)", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Total CVaR", fontsize=12)

    # Set custom ticks for every 30 steps
    step_interval = 30
    ticks = range(0, len(cvar_over_time), step_interval)
    plt.xticks(ticks, labels=[f"step_{i}" for i in ticks], rotation=45)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 🔶 2. cvar in all the nodes
    all_bus_data = CVARisk.values.flatten()
    # Create the combined distribution plot
    plt.figure(figsize=(10, 8))
    sns.histplot(all_bus_data, kde=True, bins=30, color='green', alpha=0.7)
    plt.title("Combined CVaR Distribution for All Buses", fontsize=14)
    plt.xlabel("CVaR", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 🔶 3. EMPIRICAL CDFs nodal
    plt.figure(figsize=(10, 8))
    for bus in CVARisk.index:
        sns.ecdfplot(CVARisk.loc[bus], label=bus, linewidth=1.5)
    plt.title("CVaR Distribution Across All Buses", fontsize=14)
    plt.xlabel("CVaR", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Buses", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 🔶 4. Select a specific bus for visualization
    # Set up subplots
    n_buses = len(CVARisk.index)
    n_cols = 3  # or any number you prefer
    n_rows = ceil(n_buses / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, bus_to_plot in enumerate(CVARisk.index):
        ax = axes[idx]
        bus_data = CVARisk.loc[bus_to_plot]

        sns.histplot(bus_data, kde=True, bins=10, color='blue', alpha=0.7, ax=ax)
        ax.set_title(f"{bus_to_plot}", fontsize=12)
        ax.set_xlabel("CVaR", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_comparison_CVAR_DET_SCOS(dic_res_cvar, dic_res_det, DATA):

    """ show some plots for a comparison between CVAR and DETERMINISTIC SCOScheduler"""
    names = {
        'outages': DATA['outages']['names'],  # List of outage names
        'lines': [f'line_{ll}' for ll in range(DATA['num_branches'])],  # List of line names
        'contingencies': DATA.get('n_minus1_names', None),  # List of contingency names
        'buses': [f'bus_{b}' for b in range(DATA['num_buses'])],  # List of bus names
        'generators': [f'gen_{g}' for g in DATA['network'].gen.index.tolist()],  # List of generator names
    }
    o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']

    n_col = 2
    n_rows = int(len(o_nam) / n_col)
    fig, ax = plt.subplots(n_rows, n_col, figsize=(10, n_rows * 4))
    ax = ax.flatten()
    for results_dictionary, style in zip([dic_res_det, dic_res_cvar], ['-', '--']):
        X_OutageSchedule = results_dictionary["X_OutageSchedule"]
        o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']
        # Loop through outages and plot on each axis
        for i, o in enumerate(o_nam):
            if i < len(ax):  # Ensure we don't index beyond available axes
                label = 'det-SCOP' if style == '-' else 'CVaR-SCOP'
                X_OutageSchedule.loc[o, :].plot(ax=ax[i], linestyle=style, label=label if i == 0 else None)
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(f'Outage {o}')
                ax[i].set_title(f'Outage Schedule for {o}')
                ax[i].grid()
                ax[i].legend()  # Add legend
                ax[i].tick_params(axis='x', rotation=45)  # Rotate x-tick labels

    # Hide unused axes if any
    for j in range(len(o_nam), len(ax)):
        fig.delaxes(ax[j])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle('Outage Schedule', fontsize=16)
    plt.show()

    # Plot the power generation for each generator
    n_line2_plot = min(len(l_nam), 15)
    fig, ax = plt.subplots(int(len(l_nam[:n_line2_plot]) / 3), 3, figsize=(20, 15))
    ax = ax.flatten()
    for results_dictionary, style in zip([dic_res_det, dic_res_cvar], ['-', '--']):
        Line_Flows = results_dictionary["Line_Flows"]
        for i, l in enumerate(l_nam[:n_line2_plot]):
            if i < len(ax):  # Ensure we don't index beyond available axes
                label = 'det-SCOP' if style == '-' else 'CVaR-SCOP'
                Line_Flows.loc[l, :].plot(ax=ax[i], linestyle=style, label=label if i == 0 else None)
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(l)
                ax[i].grid()
                ax[i].legend()  # Add legend
    # Adjust layout and add a title
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()




