from mpl_toolkits import mplot3d

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
#import seaborn as sns

def plot_V(dfV_pivoted):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X = np.arange(0, len(dfV_pivoted.index), 1)
    Y = np.arange(0, len(dfV_pivoted.columns), 1)
    X, Y = np.meshgrid(X, Y)
    Z = dfV_pivoted[(dfV_pivoted.index == X) & (dfV_pivoted.columns == Y)]
    
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")

    ax.set_xticks(range(0, len(dfV_pivoted.index), 2))
    ax.set_yticks(range(0, len(dfV_pivoted.columns), 2))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
            va="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="center",
            va="center", rotation_mode="anchor")
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Blues', edgecolor='none')
                  
def plot_Pi(dfPi_s_pivoted):
    #sns.heatmap(dfPi_s_pivoted, annot=True)
    fig, ax = plt.subplots()
    
    # reverse the rows in the pivoted dataset
    dfPi_s_pivoted = dfPi_s_pivoted.iloc[::-1]
    # create heatmap
    im = ax.imshow(dfPi_s_pivoted.values)
    
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")

    # set labels
    ax.set_xticks(np.arange(len(dfPi_s_pivoted.index)))
    ax.set_yticks(np.arange(len(dfPi_s_pivoted.columns)))
    ax.set_xticklabels(dfPi_s_pivoted.index[::-1])
    ax.set_yticklabels(dfPi_s_pivoted.columns[::-1])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(dfPi_s_pivoted.index)):
        for j in range(len(dfPi_s_pivoted.columns)):
            text = ax.text(j, i, np.around(dfPi_s_pivoted.iloc[i, j], decimals=2),
    ha="center", va="center", color="black")
            
    plt.show()