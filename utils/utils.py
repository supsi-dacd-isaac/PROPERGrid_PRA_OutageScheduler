from pathlib import Path
import logging
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import matplotlib.pyplot as plt

# ANSI escape code for colored text
blue_c, green_c, purple_c, cyan_c, red_c, gray_c = "\033[94m", "\033[92m", "\033[95m", "\033[96m", "\033[91m", " "
bold_c, underline_c, reset_c = "\033[1m", "\033[4m", "\033[0m"

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO)


def get_project_root() -> Path:
    """get project root path"""
    return Path(__file__).parent.parent


def plot_show(val, xlb: str = 'x', ylb: str = 'y', ax=None, **kwargs):
    """  Plot and show """
    if ax is None:
        plt.plot(val, **kwargs)
        plt.ylabel(ylb)
        plt.xlabel(xlb)
        plt.grid()
        plt.show()
    else:
        ax.plot(val, **kwargs)
        ax.set_ylabel(ylb)
        ax.set_xlabel(xlb)
        ax.grid()
        return ax


def plot_ecdf_show(val, xlb: str = 'x', ylb: str = 'ecdf', **kwargs):
    """  Plot the empirical cumulative distribution function (ECDF) for given values.   """
    # Prepare ECDF plotting
    ecdf = ECDF(val)
    x = np.sort(val)  # Sort the values for plotting
    y = ecdf(x)  # Get ECDF values for x

    plt.step(x, y, where='post', **kwargs)  # Use step plot for ECDF
    plt.ylabel(ylb)
    plt.xlabel(xlb)
    plt.grid(True)
    plt.show()

