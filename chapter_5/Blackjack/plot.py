from mpl_toolkits import mplot3d

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
#import seaborn as sns

def plot_Q(dfQ_pivoted):
    fig = plt.figure()
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