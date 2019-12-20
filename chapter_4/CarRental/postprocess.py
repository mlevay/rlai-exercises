import numpy as np
import pandas as pd

from .common import get_state_components
from .constants import DEFAULT_ACTION
from .constants import DFCOL_V_STATE_A, DFCOL_V_STATE_B, DFCOL_V_STATE, DFCOL_V_VALUE
from .common import DFCOL_PI_STATE_A, DFCOL_PI_STATE_B, DFCOL_PI_STATE, DFCOL_PI_ACTION, DFCOL_PI_PROB

def transform_data(dfPi, dfV):
    # Transform the resulting policy and value data
    dfV.iloc[:, [DFCOL_V_STATE_A, DFCOL_V_STATE_B]] = [dfV.iloc[:, DFCOL_V_STATE]].apply(get_state_components, axis=1)
    dfV_pivoted = dfV.pivot(index=DFCOL_V_STATE_A, columns=DFCOL_V_STATE_B, values=DFCOL_V_VALUE)

    groups = dict(list(dfPi.groupby(DFCOL_PI_STATE)))
    max_actions = [DEFAULT_ACTION]*len(groups)
    for s, i in zip(groups, range(len(groups))):
        dfForState = groups[s]
        max_prob = np.amax(dfForState[DFCOL_PI_PROB], axis=0)
        max_actions[i] = dfForState.loc[dfForState[DFCOL_PI_PROB] == max_prob, DFCOL_PI_ACTION].values[0]
    dfPi_s = pd.DataFrame(
        {DFCOL_PI_STATE : groups.keys(),
        DFCOL_PI_ACTION: max_actions},
        index = list(range(len(groups))))
    dfPi_s.iloc[:, [DFCOL_PI_STATE_A, DFCOL_PI_STATE_B]] = [dfPi_s.iloc[:, DFCOL_PI_STATE]].apply(get_state_components, axis=1)
    dfPi_s_pivoted = dfPi_s.pivot(index=DFCOL_PI_STATE_A, columns=DFCOL_PI_STATE_B, values=DFCOL_PI_ACTION)