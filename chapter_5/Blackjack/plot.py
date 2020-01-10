from mpl_toolkits import mplot3d

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
#import seaborn as sns

def plot_Q(dfQ_pivoted):
    fig = plt.figure(figsize=[10, 4])
    ax = plt.axes(projection='3d')

    X = dfQ_pivoted.columns.astype(int)
    Y = dfQ_pivoted.index.astype(int)
    X, Y = np.meshgrid(X, Y)
    Z = dfQ_pivoted[(dfQ_pivoted.columns == X) & (dfQ_pivoted.index == Y)]
    
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlim(-1, 1)

    ax.set_xticks(dfQ_pivoted.columns)
    ax.set_yticks(dfQ_pivoted.index)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
            va="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="center",
            va="center", rotation_mode="anchor")
    
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='Blues')
    plt.show()
    
def plot_Pi(dfPi_pivoted):
    fig, ax = plt.subplots()
    
    # reverse the rows in the pivoted dataset
    dfPi_s_pivoted = dfPi_pivoted.iloc[::-1]
    # create heatmap
    im = ax.imshow(dfPi_s_pivoted.values)
    
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")

    # set labels
    ax.set_xticks(np.arange(len(dfPi_s_pivoted.columns)))
    ax.set_yticks(np.arange(len(dfPi_s_pivoted.index)))
    ax.set_xticklabels(dfPi_s_pivoted.columns)
    ax.set_yticklabels(dfPi_s_pivoted.index)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(dfPi_s_pivoted.index)):
        for j in range(len(dfPi_s_pivoted.columns)):
            text = ax.text(j, i, np.around(dfPi_s_pivoted.iloc[i, j], decimals=2),
    ha="center", va="center", color="black")
            
    plt.show()