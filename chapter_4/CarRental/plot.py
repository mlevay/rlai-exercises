from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

def plot_V(dfV_pivoted):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X = np.arange(0, len(dfV_pivoted.index), 1)
    Y = np.arange(0, len(dfV_pivoted.columns), 1)
    X, Y = np.meshgrid(X, Y)
    Z = dfV_pivoted[(dfV_pivoted.index == X) & (dfV_pivoted.columns == Y)]

    #X = np.arange(0, 21, 1)
    #Y = np.arange(0, 21, 1)
    #X, Y = np.meshgrid(X, Y)
    #Z = np.sqrt(X**2 + Y**2)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Blues', edgecolor='none')
                  
def plot_Pi(dfPi_s_pivoted):
    pass