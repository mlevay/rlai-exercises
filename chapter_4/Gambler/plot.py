import numpy as np
import matplotlib.pyplot as plt

from .constants import MAX_ACTION, MIN_ACTION
from .constants import MAX_STATE, MIN_STATE
from .constants import PROB_HEADS


def plot_V(state_values: dict):
    scatter_plot(state_values, xlabel="Capital", ylabel="Value estimates")
    
def plot_Pi(s_a_policy: dict):
    scatter_plot(s_a_policy, xlabel="Capital", ylabel="Final policy (stake)")
    
def scatter_plot(dictionary: dict, xlabel="", ylabel=""):
    x = dictionary.keys()
    y = dictionary.values()
    
    fig, ax = plt.subplots()
    area = np.pi*3

    # Plot
    plt.scatter(x, y, s=area, alpha=0.5)
    #plt.title('Scatter plot pythonspot.com')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # set labels
    ax.set_xticks(list(range(0, MAX_STATE + 1, 25)))
    
    plt.show()