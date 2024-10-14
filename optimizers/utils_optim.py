from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import matplotlib.pyplot as plt


def plot_show(val, xlb:str ='x', ylb: str ='y', **kwargs):
    plt.plot(val, **kwargs)
    plt.ylabel(ylb)
    plt.xlabel(xlb)
    plt.grid()
    plt.show()


def plot_ecdf_show(val, xlb: str = 'x', ylb: str = 'ecdf', **kwargs):
    """
    Plot the empirical cumulative distribution function (ECDF) for given values.

    Parameters:
    - val: array-like, values to plot the ECDF for.
    - xlb: str, label for the x-axis (default is 'x').
    - ylb: str, label for the y-axis (default is 'y').
    - kwargs: additional keyword arguments for the plot.
    """
    # Prepare ECDF plotting
    ecdf = ECDF(val)
    x = np.sort(val)  # Sort the values for plotting
    y = ecdf(x)  # Get ECDF values for x

    plt.step(x, y, where='post', **kwargs)  # Use step plot for ECDF
    plt.ylabel(ylb)
    plt.xlabel(xlb)
    plt.grid(True)
    plt.show()

