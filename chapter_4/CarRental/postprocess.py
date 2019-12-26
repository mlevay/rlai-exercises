import numpy as np
import pandas as pd

from .common import get_state_components
from .constants import DEFAULT_ACTION
from .constants import DFCOL_V_STATE_A, DFCOL_V_STATE_B, DFCOL_V_STATE, DFCOL_V_VALUE
from .constants import DFCOL_PI_STATE_A, DFCOL_PI_STATE_B, DFCOL_PI_STATE, DFCOL_PI_ACTION, DFCOL_PI_PROB
from .constants import MAX_NUMBER_OF_CARS_PER_TRANSFER

def transform_data(dfPi, dfV):
    # Transform the resulting policy and value data
    dfV[[DFCOL_V_STATE_A, DFCOL_V_STATE_B]] = dfV.apply(
        lambda d: get_state_components(d[DFCOL_V_STATE]), axis=1, result_type="expand")    
    dfV_pivoted = dfV.pivot(index=DFCOL_V_STATE_A, columns=DFCOL_V_STATE_B, values=DFCOL_V_VALUE)

    groups = dict(list(dfPi.groupby(DFCOL_PI_STATE)))
    max_actions = [DEFAULT_ACTION]*len(groups)
    for s, i in zip(groups, range(len(groups))):
        dfForState = groups[s]
        max_prob = np.amax(dfForState[DFCOL_PI_PROB], axis=0)
        max_actions[i] = dfForState.loc[dfForState[DFCOL_PI_PROB] == max_prob, DFCOL_PI_ACTION].values[0]
    dfPi_s = pd.DataFrame(
        {DFCOL_PI_STATE : list(groups.keys()),
        DFCOL_PI_ACTION: max_actions},
        index = list(range(len(groups))))
    dfPi_s[[DFCOL_PI_STATE_A, DFCOL_PI_STATE_B]] = dfPi_s.apply(
        lambda d: get_state_components(d[DFCOL_PI_STATE]), axis=1, result_type="expand")
    
    # additionally, map action codes in dfPi_s_pivoted to # cars to be transferred
    num_actions = MAX_NUMBER_OF_CARS_PER_TRANSFER*2 + 1 # all whole numbers in [-n, n]
    all_actions = np.array([list(range(-MAX_NUMBER_OF_CARS_PER_TRANSFER, MAX_NUMBER_OF_CARS_PER_TRANSFER+1))])
    action_names = np.array([0] * num_actions)
    for i in range(num_actions):
        action_name = i
        action_names[i] = action_name
    all_actions = np.hstack((all_actions.T, np.atleast_2d(action_names).T))
    dict_actions = dict(zip(all_actions[:,-1], all_actions[:,0]))   
    dfPi_s[DFCOL_PI_ACTION] = dfPi_s[DFCOL_PI_ACTION].map(dict_actions)  
    
    dfPi_s_pivoted = dfPi_s.pivot(index=DFCOL_PI_STATE_A, columns=DFCOL_PI_STATE_B, values=DFCOL_PI_ACTION)
    
    return dfV_pivoted, dfPi_s_pivoted